//! Partitioning for DuckLake tables: splitting a write into one data file per
//! distinct partition value, and pruning whole files at scan time when a query
//! predicate excludes their partition value.
//!
//! Identity and the monotonic temporal transforms (year / month / day / hour) are
//! modelled. A file records its partition value — the value its rows share for
//! each partition column, after the transform — so a predicate on a partition
//! column can skip a file whenever the value provably fails it. Because the
//! temporal transforms are monotonic, pruning is the same comparison applied to
//! `transform(value)` vs `transform(literal)`: identity compares the exact value
//! (`=`, `IN`, `<`, `<=`, `>`, `>=`, `BETWEEN`), and a temporal transform compares
//! the coarser bucket conservatively (a range predicate keeps the boundary
//! bucket). Hash (`bucket`) and `truncate` transforms are out of scope.

use std::collections::HashMap;

use arrow::datatypes::DataType;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::{Expr, Operator};

#[cfg(feature = "write")]
use crate::Result;
#[cfg(feature = "write")]
use arrow::array::{RecordBatch, UInt32Array};
#[cfg(feature = "write")]
use arrow::compute::{concat_batches, take_record_batch};
#[cfg(feature = "write")]
use arrow::datatypes::SchemaRef;

/// A DuckLake partition transform. Identity and the monotonic temporal transforms
/// are supported; hash (`bucket`) and `truncate` transforms are not (they need
/// different pruning semantics and are out of scope).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PartitionTransform {
    /// Partition by the column's exact value.
    Identity,
    /// Partition by calendar year of a date/timestamp column.
    Year,
    /// Partition by calendar month of a date/timestamp column.
    Month,
    /// Partition by calendar day of a date/timestamp column.
    Day,
    /// Partition by hour of a timestamp column.
    Hour,
}

impl PartitionTransform {
    /// The DuckLake catalog `transform` string for this transform.
    pub fn as_catalog_str(self) -> &'static str {
        match self {
            PartitionTransform::Identity => "identity",
            PartitionTransform::Year => "year",
            PartitionTransform::Month => "month",
            PartitionTransform::Day => "day",
            PartitionTransform::Hour => "hour",
        }
    }

    /// Parse the DuckLake catalog `transform` string. Unknown / unsupported
    /// transforms return `None`.
    pub fn from_catalog_str(s: &str) -> Option<Self> {
        match s {
            "identity" => Some(PartitionTransform::Identity),
            "year" => Some(PartitionTransform::Year),
            "month" => Some(PartitionTransform::Month),
            "day" => Some(PartitionTransform::Day),
            "hour" => Some(PartitionTransform::Hour),
            _ => None,
        }
    }

    /// Whether this transform derives its partition value from a temporal column.
    pub fn is_temporal(self) -> bool {
        !matches!(self, PartitionTransform::Identity)
    }
}

/// One partition key: a table column and the transform applied to it.
#[derive(Debug, Clone)]
pub struct PartitionColumn {
    /// Name of the table column this key partitions on.
    pub column_name: String,
    /// Transform applied to the column value to derive the partition value.
    pub transform: PartitionTransform,
}

/// A table's partition spec — an ordered list of partition keys. An empty spec
/// means the table is not partitioned.
#[derive(Debug, Clone, Default)]
pub struct PartitionSpec {
    /// The partition keys, in key order.
    pub columns: Vec<PartitionColumn>,
}

impl PartitionSpec {
    /// Build an identity partition spec over the given column names, in order.
    pub fn identity<I, S>(columns: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        Self {
            columns: columns
                .into_iter()
                .map(|c| PartitionColumn {
                    column_name: c.into(),
                    transform: PartitionTransform::Identity,
                })
                .collect(),
        }
    }

    /// Whether the spec has no partition keys.
    pub fn is_empty(&self) -> bool {
        self.columns.is_empty()
    }
}

/// Apply a partition transform to a value, yielding the partition value that is
/// stored (on write) and compared (on prune). Identity returns the value
/// unchanged; a temporal transform returns a monotonic `Int64` — the year number,
/// months-since-1970, days-since-epoch, or hours-since-epoch. Returns `None` for a
/// NULL value or a non-temporal value under a temporal transform.
pub(crate) fn apply_transform(
    transform: PartitionTransform,
    value: &ScalarValue,
) -> Option<ScalarValue> {
    if matches!(transform, PartitionTransform::Identity) {
        return (!value.is_null()).then(|| value.clone());
    }
    let seconds = epoch_seconds(value)?;
    let days = seconds.div_euclid(86_400);
    let bucket = match transform {
        PartitionTransform::Identity => return None,
        PartitionTransform::Year => civil_from_days(days).0,
        PartitionTransform::Month => {
            let (year, month, _) = civil_from_days(days);
            year * 12 + (month - 1)
        },
        PartitionTransform::Day => days,
        PartitionTransform::Hour => seconds.div_euclid(3_600),
    };
    Some(ScalarValue::Int64(Some(bucket)))
}

/// Seconds since the Unix epoch for a date/timestamp scalar; `None` for a NULL or
/// non-temporal value.
fn epoch_seconds(value: &ScalarValue) -> Option<i64> {
    Some(match value {
        ScalarValue::Date32(Some(days)) => (*days as i64) * 86_400,
        ScalarValue::Date64(Some(ms)) => ms.div_euclid(1_000),
        ScalarValue::TimestampSecond(Some(s), _) => *s,
        ScalarValue::TimestampMillisecond(Some(ms), _) => ms.div_euclid(1_000),
        ScalarValue::TimestampMicrosecond(Some(us), _) => us.div_euclid(1_000_000),
        ScalarValue::TimestampNanosecond(Some(ns), _) => ns.div_euclid(1_000_000_000),
        _ => return None,
    })
}

/// `(year, month, day)` from days since 1970-01-01 in the proleptic Gregorian
/// calendar (Howard Hinnant's `civil_from_days`).
fn civil_from_days(z: i64) -> (i64, i64, i64) {
    let z = z + 719_468;
    let era = if z >= 0 {
        z
    } else {
        z - 146_096
    }
    .div_euclid(146_097);
    let doe = z - era * 146_097;
    let yoe = (doe - doe / 1_460 + doe / 36_524 - doe / 146_096) / 365;
    let year = yoe + era * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let day = doy - (153 * mp + 2) / 5 + 1;
    let month = if mp < 10 {
        mp + 3
    } else {
        mp - 9
    };
    (
        if month <= 2 {
            year + 1
        } else {
            year
        },
        month,
        day,
    )
}

/// Whether `data_type` is a date/timestamp type a temporal transform can read.
#[cfg(feature = "write")]
pub(crate) fn is_temporal_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::Date32 | DataType::Date64 | DataType::Timestamp(_, _)
    )
}

/// One partitioned output group: the shared partition value of each partition
/// column (in key order, `None` = NULL) and the rows carrying it.
#[cfg(feature = "write")]
pub(crate) struct PartitionGroup {
    /// Partition value of each partition column, in key order (post-transform).
    pub key: Vec<Option<ScalarValue>>,
    /// The rows sharing this key, as a single batch.
    pub batch: RecordBatch,
}

/// Split `batches` into one concatenated batch per distinct partition-value tuple.
///
/// `partition_indices` are the physical column positions of the partition columns
/// and `transforms` their transforms, both in key order. Every returned group's
/// rows share the same transformed partition value for each of those columns.
#[cfg(feature = "write")]
pub(crate) fn split_by_partition(
    schema: &SchemaRef,
    batches: &[RecordBatch],
    partition_indices: &[usize],
    transforms: &[PartitionTransform],
) -> Result<Vec<PartitionGroup>> {
    let batch = concat_batches(schema, batches)?;
    let num_rows = batch.num_rows();

    // Group row positions by their transformed partition-value tuple, preserving
    // first-seen order so the output is deterministic across runs.
    let mut order: Vec<Vec<Option<ScalarValue>>> = Vec::new();
    let mut rows_by_key: HashMap<Vec<Option<ScalarValue>>, Vec<u32>> = HashMap::new();
    for row in 0..num_rows {
        let key = partition_indices
            .iter()
            .zip(transforms)
            .map(|(&col, &transform)| {
                let raw = ScalarValue::try_from_array(batch.column(col), row)?;
                Ok(apply_transform(transform, &raw))
            })
            .collect::<Result<Vec<_>>>()?;
        if !rows_by_key.contains_key(&key) {
            order.push(key.clone());
        }
        rows_by_key.entry(key).or_default().push(row as u32);
    }

    let mut groups = Vec::with_capacity(order.len());
    for key in order {
        let rows = rows_by_key.remove(&key).unwrap_or_default();
        let indices = UInt32Array::from(rows);
        let part = take_record_batch(&batch, &indices)?;
        groups.push(PartitionGroup {
            key,
            batch: part,
        });
    }
    Ok(groups)
}

/// Render a partition value as the catalog `partition_value` string.
#[cfg(feature = "write")]
pub(crate) fn scalar_to_catalog_value(value: &ScalarValue) -> Option<String> {
    if value.is_null() {
        return None;
    }
    match value.cast_to(&DataType::Utf8) {
        Ok(ScalarValue::Utf8(Some(s))) => Some(s),
        _ => Some(value.to_string()),
    }
}

/// Parse a catalog `partition_value` string into the scalar the pruner compares.
/// Temporal partition values are stored as monotonic `Int64`; identity values
/// keep the partition column's type. A NULL (`None`) or unparseable value yields
/// `None`, which the pruner treats conservatively (never skips the file).
pub(crate) fn catalog_value_to_scalar(
    value: Option<&str>,
    transform: PartitionTransform,
    data_type: &DataType,
) -> Option<ScalarValue> {
    let raw = value?;
    let target = if transform.is_temporal() {
        &DataType::Int64
    } else {
        data_type
    };
    ScalarValue::try_from_string(raw.to_string(), target).ok()
}

/// A file's partition value for one column, with the transform that produced it —
/// enough to decide, for a predicate literal, whether the file can be skipped.
pub(crate) struct FilePartitionValue {
    /// The transform applied to the partition column.
    pub transform: PartitionTransform,
    /// The file's partition value (post-transform), or `None` for a NULL partition
    /// (never pruned).
    pub value: Option<ScalarValue>,
}

/// Whether a file whose partition columns hold `values` could contain a row
/// matching every filter. Returns `false` only when some filter *provably*
/// excludes the file, so the caller may safely skip it; DataFusion re-applies the
/// filters after the scan, so a conservative `true` is always correct.
pub(crate) fn file_matches_filters(
    values: &HashMap<String, FilePartitionValue>,
    filters: &[Expr],
) -> bool {
    !filters.iter().any(|f| excludes(values, f))
}

/// Whether `filter` proves no row in a file with these partition `values` can
/// match. Only decidable partition-column predicates return `true`; anything else
/// (non-partition columns, NULL values, unknown shapes) returns `false`.
fn excludes(values: &HashMap<String, FilePartitionValue>, filter: &Expr) -> bool {
    match filter {
        Expr::BinaryExpr(be) => match be.op {
            Operator::And => excludes(values, &be.left) || excludes(values, &be.right),
            Operator::Or => excludes(values, &be.left) && excludes(values, &be.right),
            op => comparison_excludes(values, &be.left, op, &be.right),
        },
        Expr::Between(b) => {
            let Some(fpv) = partition_value(values, &b.expr) else {
                return false;
            };
            let (Some(low), Some(high)) = (as_literal(&b.low), as_literal(&b.high)) else {
                return false;
            };
            let within = survives(fpv, Operator::GtEq, &low)
                .and_then(|ge| survives(fpv, Operator::LtEq, &high).map(|le| ge && le));
            match within {
                Some(inside) => b.negated == inside,
                None => false,
            }
        },
        Expr::InList(inlist) => {
            let Some(fpv) = partition_value(values, &inlist.expr) else {
                return false;
            };
            let mut found = false;
            for item in &inlist.list {
                let Some(lit) = as_literal(item) else {
                    return false;
                };
                match survives(fpv, Operator::Eq, &lit) {
                    Some(true) => found = true,
                    Some(false) => {},
                    None => return false,
                }
            }
            inlist.negated == found
        },
        _ => false,
    }
}

/// Evaluate a `left OP right` comparison where exactly one side is a partition
/// column and the other a literal. Returns `true` only when the file's value
/// provably fails the comparison.
fn comparison_excludes(
    values: &HashMap<String, FilePartitionValue>,
    left: &Expr,
    op: Operator,
    right: &Expr,
) -> bool {
    let (fpv, op, literal) = if let Some(fpv) = partition_value(values, left) {
        match as_literal(right) {
            Some(lit) => (fpv, op, lit),
            None => return false,
        }
    } else if let Some(fpv) = partition_value(values, right) {
        match as_literal(left) {
            // Flip so the partition value stays on the left of the operator.
            Some(lit) => (fpv, swap_operator(op), lit),
            None => return false,
        }
    } else {
        return false;
    };

    matches!(survives(fpv, op, &literal), Some(false))
}

/// The file's partition value for `expr` when `expr` is a partition column;
/// `None` otherwise.
fn partition_value<'a>(
    values: &'a HashMap<String, FilePartitionValue>,
    expr: &Expr,
) -> Option<&'a FilePartitionValue> {
    let Expr::Column(col) = expr else {
        return None;
    };
    values.get(&col.name)
}

fn as_literal(expr: &Expr) -> Option<ScalarValue> {
    match expr {
        Expr::Literal(scalar, ..) => Some(scalar.clone()),
        _ => None,
    }
}

/// Whether a file with partition value `fpv` could still satisfy `value OP literal`.
///
/// For identity the comparison is exact (`fpv.value OP literal`). For a temporal
/// transform both sides are projected through the (monotonic) transform and the
/// comparison is *conservative*: a range predicate keeps the boundary bucket
/// (`Lt`/`LtEq` → `bucket <= transform(lit)`, `Gt`/`GtEq` → `bucket >=
/// transform(lit)`), and `!=` never prunes (the bucket spans values other than
/// `lit`). `None` when the literal is not comparable — the file is kept.
fn survives(fpv: &FilePartitionValue, op: Operator, literal: &ScalarValue) -> Option<bool> {
    let value = fpv.value.as_ref()?;
    let exact = matches!(fpv.transform, PartitionTransform::Identity);
    let target = if exact {
        literal.cast_to(&value.data_type()).ok()?
    } else {
        apply_transform(fpv.transform, literal)?
    };
    match op {
        Operator::Eq => Some(value == &target),
        Operator::NotEq => Some(if exact {
            value != &target
        } else {
            true
        }),
        Operator::Lt => value.partial_cmp(&target).map(|o| {
            if exact {
                o.is_lt()
            } else {
                o.is_le()
            }
        }),
        Operator::LtEq => value.partial_cmp(&target).map(|o| o.is_le()),
        Operator::Gt => value.partial_cmp(&target).map(|o| {
            if exact {
                o.is_gt()
            } else {
                o.is_ge()
            }
        }),
        Operator::GtEq => value.partial_cmp(&target).map(|o| o.is_ge()),
        _ => None,
    }
}

/// Mirror a comparison operator so `literal OP column` becomes `column OP' literal`.
fn swap_operator(op: Operator) -> Operator {
    match op {
        Operator::Lt => Operator::Gt,
        Operator::LtEq => Operator::GtEq,
        Operator::Gt => Operator::Lt,
        Operator::GtEq => Operator::LtEq,
        other => other,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn int64(v: i64) -> ScalarValue {
        ScalarValue::Int64(Some(v))
    }

    #[test]
    fn civil_from_days_matches_known_dates() {
        assert_eq!(civil_from_days(0), (1970, 1, 1));
        assert_eq!(civil_from_days(31), (1970, 2, 1)); // Jan 1970 has 31 days
        assert_eq!(civil_from_days(59), (1970, 3, 1)); // + Feb 1970's 28 days
        assert_eq!(civil_from_days(365), (1971, 1, 1));
    }

    #[test]
    fn apply_transform_on_date32() {
        // Date32 is days since epoch; month/year/day buckets are monotonic ints.
        let jan = ScalarValue::Date32(Some(15)); // 1970-01-16
        let mar = ScalarValue::Date32(Some(59)); // 1970-03-01
        assert_eq!(
            apply_transform(PartitionTransform::Year, &jan),
            Some(int64(1970))
        );
        assert_eq!(
            apply_transform(PartitionTransform::Month, &jan),
            Some(int64(1970 * 12)) // 1970-01
        );
        assert_eq!(
            apply_transform(PartitionTransform::Month, &mar),
            Some(int64(1970 * 12 + 2)) // 1970-03
        );
        assert_eq!(
            apply_transform(PartitionTransform::Day, &jan),
            Some(int64(15))
        );
        assert_eq!(
            apply_transform(PartitionTransform::Hour, &jan),
            Some(int64(15 * 24))
        );
        // Identity keeps the value; a non-temporal value has no temporal bucket.
        assert_eq!(
            apply_transform(PartitionTransform::Identity, &jan),
            Some(ScalarValue::Date32(Some(15)))
        );
        assert_eq!(
            apply_transform(
                PartitionTransform::Month,
                &ScalarValue::Utf8(Some("x".into()))
            ),
            None
        );
    }

    #[test]
    fn month_transform_prunes_conservatively() {
        // A Feb-1970 file (month bucket 1970*12+1) against ts predicates.
        let feb = FilePartitionValue {
            transform: PartitionTransform::Month,
            value: Some(int64(1970 * 12 + 1)),
        };
        let mar1 = ScalarValue::Date32(Some(59)); // 1970-03-01, month bucket +2
        let feb15 = ScalarValue::Date32(Some(45)); // 1970-02-15, month bucket +1

        // ts >= Mar 1 excludes the Feb file (bucket +1 < +2).
        assert_eq!(survives(&feb, Operator::GtEq, &mar1), Some(false));
        // ts >= a date inside Feb keeps the Feb file (boundary bucket).
        assert_eq!(survives(&feb, Operator::GtEq, &feb15), Some(true));
        // ts = a date inside Feb keeps the Feb file; = a March date excludes it.
        assert_eq!(survives(&feb, Operator::Eq, &feb15), Some(true));
        assert_eq!(survives(&feb, Operator::Eq, &mar1), Some(false));
        // != never prunes a temporal bucket (it spans other values).
        assert_eq!(survives(&feb, Operator::NotEq, &feb15), Some(true));
    }
}
