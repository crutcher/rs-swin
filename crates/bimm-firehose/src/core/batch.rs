use crate::core::{Row, TableSchema};
use std::ops::{Index, IndexMut};
use std::sync::Arc;

/// Represents a batch of `BimmRow`, with `BimmTableSchema` as its schema.
#[derive(Debug)]
pub struct RowBatch {
    /// The schema of the table slice.
    pub schema: Arc<TableSchema>,

    /// The rows in the table slice.
    pub rows: Vec<Row>,
}

impl Index<usize> for RowBatch {
    type Output = Row;

    /// Returns a reference to the row at the specified index.
    ///
    /// ## Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for RowBatch {
    /// Returns a mutable reference to the row at the specified index.
    ///
    /// ## Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index_mut(
        &mut self,
        index: usize,
    ) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl RowBatch {
    /// Creates a new `BimmTableSlice` with the given schema and rows.
    ///
    /// ## Arguments
    ///
    /// * `schema`: The schema of the table slice.
    /// * `rows`: The rows in the table slice.
    pub fn new(
        schema: Arc<TableSchema>,
        rows: Vec<Row>,
    ) -> Self {
        RowBatch { schema, rows }
    }

    /// Creates a new `BimmRowBatch` with the given schema and a specified size.
    ///
    /// This initializes the batch with empty rows, each having the width of the schema.
    ///
    /// ## Arguments
    ///
    /// * `schema`: The schema of the table slice.
    /// * `size`: The number of rows to initialize in the batch.
    pub fn with_size(
        schema: Arc<TableSchema>,
        size: usize,
    ) -> Self {
        let width = schema.columns.len();
        let rows = vec![Row::new_with_width(width); size];

        RowBatch { schema, rows }
    }

    /// Returns the number of rows in the table slice.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Checks if the table slice is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Takes a sub-slice of the table slice from `start` to `end`.
    ///
    /// TODO: use Burn's new `AsIndex` trait support when available.
    ///
    /// ## Arguments
    ///
    /// * `start`: The starting index of the slice (inclusive).
    /// * `end`: The ending index of the slice (exclusive).
    ///
    /// ## Returns
    ///
    /// A new `BimmTableSlice` containing the rows from `start` to `end`.
    pub fn slice(
        &self,
        start: usize,
        end: usize,
    ) -> Self {
        RowBatch::new(self.schema.clone(), self.rows[start..end].to_vec())
    }

    /// Collects values from a specified column by its name.
    ///
    /// Aggregates over `row.get_value::<T>(index)` for each row in the table slice.
    ///
    /// ## Arguments
    ///
    /// * `column_name`: The name of the column to collect values from.
    ///
    /// ## Returns
    ///
    /// A vector of `Option<&T>` where `T` is the type of the column values.
    pub fn collect_column_values<T: 'static>(
        &self,
        column_name: &str,
    ) -> Vec<Option<&T>> {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.rows
            .iter()
            .map(|row| row.get_slot::<T>(index))
            .collect::<Vec<_>>()
    }

    /// Joins values from a specified column by its name using a custom function.
    ///
    /// The function `join_fn` is applied to the collected values of the column.
    ///
    /// ## Arguments
    ///
    /// * `column_name`: The name of the column to join values from.
    /// * `join_fn`: A function that takes a slice of `Option<&T>` and returns a `Result<V, String>`.
    ///
    /// ## Returns
    ///
    /// A `Result<V, String>` where `V` is the type returned by the `join_fn`.
    pub fn join_column<T: 'static, V>(
        &self,
        column_name: &str,
        join_fn: impl Fn(&Vec<Option<&T>>) -> Result<V, String>,
    ) -> Result<V, String> {
        let values = self.collect_column_values::<T>(column_name);
        join_fn(&values)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::{ColumnSchema, Row, TableSchema};
    use std::sync::Arc;

    #[test]
    fn test_with_size() {
        let schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let batch = RowBatch::with_size(Arc::new(schema), 3);
        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_row_batch_with_basic_types() {
        let schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let row1 = Row::new_with_columns(
            &schema,
            &["foo", "bar"],
            [Arc::new(42), Arc::new("Hello".to_string())],
        );

        let row2 = Row::new_with_columns(
            &schema,
            &["foo", "bar"],
            [Arc::new(100), Arc::new("World".to_string())],
        );

        let batch = RowBatch::new(Arc::new(schema), vec![row1, row2]);

        assert_eq!(batch.len(), 2);
        assert!(!batch.is_empty());

        assert_eq!(batch.rows[0].get_slot::<i32>(0), Some(&42));
        assert_eq!(
            batch.rows[1].get_slot::<String>(1),
            Some(&"World".to_string())
        );

        assert_eq!(
            batch.collect_column_values::<i32>("foo"),
            vec![Some(&42), Some(&100)]
        );
        assert_eq!(
            batch.join_column("foo", |vs| {
                let x: i32 = vs.iter().filter_map(|&v| v).sum();
                Ok(x)
            }),
            Ok(142_i32),
        );
    }

    #[test]
    fn test_row_batch() {
        let schema = TableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);

        let batch = RowBatch::new(
            Arc::new(schema.clone()),
            vec![
                Row::new_with_columns(&schema, &["foo"], [Arc::new(1)]),
                Row::new_with_columns(&schema, &["foo"], [Arc::new(2)]),
                Row::new_with_columns(&schema, &["foo"], [Arc::new(3)]),
                Row::new_with_columns(&schema, &["foo"], [Arc::new(4)]),
            ],
        );

        let part = batch.slice(1, 3);
        assert_eq!(part.len(), 2);
        assert_eq!(part[0].get_slot::<i32>(0), Some(&2));
        assert_eq!(part[1].get_slot::<i32>(0), Some(&3));
    }

    #[test]
    fn test_mut_row_batch() {
        let schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let mut batch = RowBatch::new(
            Arc::new(schema.clone()),
            vec![
                Row::new_with_columns(&schema, &["foo"], [Arc::new(1)]),
                Row::new_with_columns(&schema, &["foo"], [Arc::new(2)]),
            ],
        );

        batch[0].set_columns(&schema, &["bar"], [Arc::new("Hello".to_string())]);
        batch[1].set_columns(&schema, &["bar"], [Arc::new("World".to_string())]);

        assert_eq!(
            batch.join_column("bar", |vs: &Vec<Option<&String>>| {
                let joined = vs
                    .iter()
                    .filter_map(|&v| v)
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ");
                Ok(joined)
            }),
            Ok("Hello, World".to_string())
        );
    }
}
