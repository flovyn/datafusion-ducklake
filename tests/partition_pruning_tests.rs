//! Integration tests for identity partitioning: partitioned writes lay out one
//! data file per partition value, record those values, and let a scan skip whole
//! files whose value a predicate excludes — while still reading back the same
//! rows a full scan would.

#![cfg(all(feature = "write-sqlite", feature = "metadata-sqlite"))]

use std::sync::Arc;

use arrow::array::{Date32Array, Int32Array, Int64Array, StringArray};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::{CatalogProvider, TableProvider};
use datafusion::common::ScalarValue;
use datafusion::prelude::*;
use object_store::local::LocalFileSystem;
use sqlx::Row;
use tempfile::TempDir;

use datafusion_ducklake::{
    DuckLakeCatalog, DuckLakeTable, DuckLakeTableWriter, MetadataWriter, PartitionColumn,
    PartitionSpec, PartitionTransform, SqliteMetadataProvider, SqliteMetadataWriter,
};

async fn create_test_env() -> (SqliteMetadataWriter, TempDir) {
    let temp_dir = TempDir::new().unwrap();
    let db_path = temp_dir.path().join("test.db");
    let data_path = temp_dir.path().join("data");
    std::fs::create_dir_all(&data_path).unwrap();

    let conn_str = format!("sqlite:{}?mode=rwc", db_path.display());
    let writer = SqliteMetadataWriter::new_with_init(&conn_str)
        .await
        .unwrap();
    writer.set_data_path(data_path.to_str().unwrap()).unwrap();

    (writer, temp_dir)
}

async fn read_catalog(temp_dir: &TempDir) -> DuckLakeCatalog {
    let db_path = temp_dir.path().join("test.db");
    let conn_str = format!("sqlite:{}", db_path.display());
    let provider = SqliteMetadataProvider::new(&conn_str).await.unwrap();
    DuckLakeCatalog::new(provider).unwrap()
}

fn orders_schema() -> Arc<Schema> {
    Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("region", DataType::Utf8, false),
        Field::new("amount", DataType::Int64, false),
    ]))
}

/// id / region / amount rows spread across three regions (US, EU, AP), so an
/// identity partition on `region` yields exactly three files.
fn orders_batch() -> RecordBatch {
    let schema = orders_schema();
    RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(StringArray::from(vec!["US", "EU", "AP", "US", "AP"])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400, 500])),
        ],
    )
    .unwrap()
}

async fn ducklake_table(catalog: &DuckLakeCatalog, table: &str) -> Arc<dyn TableProvider> {
    catalog
        .schema("main")
        .unwrap()
        .table(table)
        .await
        .unwrap()
        .unwrap()
}

#[tokio::test(flavor = "multi_thread")]
async fn partitioned_write_lays_out_one_file_per_partition_value() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = Arc::new(LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();

    let result = table_writer
        .write_table_partitioned(
            "main",
            "orders",
            &[orders_batch()],
            &PartitionSpec::identity(["region"]),
        )
        .await
        .unwrap();

    assert_eq!(result.records_written, 5);
    assert_eq!(result.files_written, 3, "one data file per region");

    // Every row reads back, unchanged, on a fresh context.
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(read_catalog(&temp_dir).await));
    let rows = ctx
        .sql("SELECT id, region, amount FROM test.main.orders ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let total: usize = rows.iter().map(|b| b.num_rows()).sum();
    assert_eq!(total, 5);

    // The catalog records each file's exact partition value.
    let conn_str = format!("sqlite:{}", temp_dir.path().join("test.db").display());
    let pool = sqlx::SqlitePool::connect(&conn_str).await.unwrap();
    let value_rows = sqlx::query(
        "SELECT partition_value, COUNT(*) AS n
         FROM ducklake_file_partition_value
         GROUP BY partition_value ORDER BY partition_value",
    )
    .fetch_all(&pool)
    .await
    .unwrap();
    let recorded: Vec<(String, i64)> = value_rows
        .iter()
        .map(|r| (r.get::<String, _>("partition_value"), r.get::<i64, _>("n")))
        .collect();
    assert_eq!(
        recorded,
        vec![("AP".to_string(), 1), ("EU".to_string(), 1), ("US".to_string(), 1),],
        "exactly one file recorded per region value"
    );

    // The spec is recorded as an identity partition on `region`.
    let transform: String = sqlx::query_scalar(
        "SELECT transform FROM ducklake_partition_column WHERE partition_key_index = 0",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(transform, "identity");
}

#[tokio::test(flavor = "multi_thread")]
async fn equality_predicate_skips_whole_files_by_partition_value() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = Arc::new(LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table_partitioned(
            "main",
            "orders",
            &[orders_batch()],
            &PartitionSpec::identity(["region"]),
        )
        .await
        .unwrap();

    let catalog = read_catalog(&temp_dir).await;
    let provider = ducklake_table(&catalog, "orders").await;
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("read path builds a DuckLakeTable");

    let all = table.plan_file_count(&[]);
    assert_eq!(all, 3, "unpartitioned scan opens every region file");

    let eq = col("region").eq(lit("US"));
    assert_eq!(
        table.plan_file_count(std::slice::from_ref(&eq)),
        1,
        "region = 'US' opens only the US file"
    );
    assert!(table.plan_file_count(std::slice::from_ref(&eq)) < all);

    // A range predicate prunes too: 'AP' < 'E' is skipped, 'EU'/'US' kept.
    let range = col("region").gt(lit("E"));
    assert_eq!(table.plan_file_count(std::slice::from_ref(&range)), 2);

    // IN-list pruning.
    let in_list = col("region").in_list(vec![lit("EU"), lit("AP")], false);
    assert_eq!(table.plan_file_count(std::slice::from_ref(&in_list)), 2);

    // A predicate on a non-partition column prunes nothing (correctly).
    let non_part = col("amount").gt(lit(150_i64));
    assert_eq!(table.plan_file_count(std::slice::from_ref(&non_part)), 3);
}

#[tokio::test(flavor = "multi_thread")]
async fn month_transform_lays_out_and_prunes_by_month() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = Arc::new(LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();

    // Date32 days in 1970: Jan {0, 15}, Feb {31, 45}, Mar {59} → 3 month buckets.
    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("ts", DataType::Date32, false),
        Field::new("amount", DataType::Int64, false),
    ]));
    let batch = RecordBatch::try_new(
        schema,
        vec![
            Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5])),
            Arc::new(Date32Array::from(vec![0, 31, 59, 15, 45])),
            Arc::new(Int64Array::from(vec![100, 200, 300, 400, 500])),
        ],
    )
    .unwrap();

    let spec = PartitionSpec {
        columns: vec![PartitionColumn {
            column_name: "ts".to_string(),
            transform: PartitionTransform::Month,
        }],
    };
    let result = table_writer
        .write_table_partitioned("main", "events", &[batch], &spec)
        .await
        .unwrap();
    assert_eq!(result.files_written, 3, "one data file per month");

    let catalog = read_catalog(&temp_dir).await;
    let provider = ducklake_table(&catalog, "events").await;
    let table = (provider.as_ref() as &dyn std::any::Any)
        .downcast_ref::<DuckLakeTable>()
        .expect("read path builds a DuckLakeTable");

    assert_eq!(table.plan_file_count(&[]), 3);

    // ts >= 1970-03-01 keeps only the March bucket (conservative, monotonic).
    let ge_mar = col("ts").gt_eq(lit(ScalarValue::Date32(Some(59))));
    assert_eq!(table.plan_file_count(std::slice::from_ref(&ge_mar)), 1);
    // ts >= 1970-02-01 keeps Feb + Mar.
    let ge_feb = col("ts").gt_eq(lit(ScalarValue::Date32(Some(31))));
    assert_eq!(table.plan_file_count(std::slice::from_ref(&ge_feb)), 2);
    // ts = a January date keeps only the January bucket.
    let eq_jan = col("ts").eq(lit(ScalarValue::Date32(Some(15))));
    assert_eq!(table.plan_file_count(std::slice::from_ref(&eq_jan)), 1);

    // Pruned results stay correct: ts >= Mar 1 returns exactly the March row.
    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(read_catalog(&temp_dir).await));
    let rows = ctx
        .sql("SELECT id FROM test.main.events WHERE ts >= DATE '1970-03-01' ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids = rows[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    assert_eq!(ids.values(), &[3]);
}

#[tokio::test(flavor = "multi_thread")]
async fn pruned_query_matches_full_scan_baseline() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = Arc::new(LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();
    table_writer
        .write_table_partitioned(
            "main",
            "orders",
            &[orders_batch()],
            &PartitionSpec::identity(["region"]),
        )
        .await
        .unwrap();

    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(read_catalog(&temp_dir).await));

    // Pruned scan (partition predicate) vs the exact rows a full scan filters to.
    let pruned = ctx
        .sql("SELECT id, amount FROM test.main.orders WHERE region = 'US' ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ids = pruned[0]
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap();
    let amounts = pruned[0]
        .column(1)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap();
    assert_eq!(ids.values(), &[1, 4]);
    assert_eq!(amounts.values(), &[100, 400]);
}

#[tokio::test(flavor = "multi_thread")]
async fn append_partitioned_keeps_existing_files_and_lands_new_partition_values() {
    let (writer, temp_dir) = create_test_env().await;
    let object_store = Arc::new(LocalFileSystem::new());
    let table_writer = DuckLakeTableWriter::new(Arc::new(writer), object_store).unwrap();

    // Baseline: 5 rows across US/EU/AP → three data files.
    table_writer
        .write_table_partitioned(
            "main",
            "orders",
            &[orders_batch()],
            &PartitionSpec::identity(["region"]),
        )
        .await
        .unwrap();

    // Append two US rows and one new-region LA row: one new file per appended
    // distinct value (US, LA), the three baseline files left in place.
    let append = RecordBatch::try_new(
        orders_schema(),
        vec![
            Arc::new(Int32Array::from(vec![6, 7, 8])),
            Arc::new(StringArray::from(vec!["US", "US", "LA"])),
            Arc::new(Int64Array::from(vec![600, 700, 800])),
        ],
    )
    .unwrap();
    let result = table_writer
        .append_table_partitioned(
            "main",
            "orders",
            &[append],
            &PartitionSpec::identity(["region"]),
        )
        .await
        .unwrap();
    assert_eq!(result.records_written, 3);
    assert_eq!(result.files_written, 2, "one appended file per distinct region (US, LA)");

    // Append keeps the prior generation: 3 baseline + 2 appended = 5 live files
    // (a Replace would have retired the baseline three).
    let conn_str = format!("sqlite:{}", temp_dir.path().join("test.db").display());
    let pool = sqlx::SqlitePool::connect(&conn_str).await.unwrap();
    let live_files: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_data_file WHERE end_snapshot IS NULL",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(live_files, 5, "baseline files kept, appended files added");

    // US now spans two files (one baseline, one appended); every row reads back.
    let us_files: i64 = sqlx::query_scalar(
        "SELECT COUNT(*) FROM ducklake_file_partition_value WHERE partition_value = 'US'",
    )
    .fetch_one(&pool)
    .await
    .unwrap();
    assert_eq!(us_files, 2, "US spans a baseline and an appended file");

    let ctx = SessionContext::new();
    ctx.register_catalog("test", Arc::new(read_catalog(&temp_dir).await));
    let rows = ctx
        .sql("SELECT id FROM test.main.orders ORDER BY id")
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let ids: Vec<i32> = rows
        .iter()
        .flat_map(|b| {
            b.column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .unwrap()
                .values()
                .to_vec()
        })
        .collect();
    assert_eq!(ids, vec![1, 2, 3, 4, 5, 6, 7, 8], "baseline + appended rows");
}
