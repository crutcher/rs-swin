use crate::core::schema::FirehoseTableSchema;
use std::any::Any;
use std::ops::{Index, IndexMut};
use std::sync::Arc;

/// Represents a boxed value that can hold any type.
pub type AnyArc = Arc<dyn Any>;

/// Represents a row in a Bimm table, containing values for each column.
#[derive(Debug, Clone)]
pub struct Row {
    /// The values in the row, where each value is an `Option<AnyArc>`.
    pub slots: Vec<Option<AnyArc>>,
}

impl Row {
    /// Creates a new `BurnRow` with the given values.
    pub fn new_with_width(size: usize) -> Self {
        let mut slots = Vec::with_capacity(size);
        slots.resize_with(size, || None);

        Row { slots }
    }

    /// Creates an empty `BurnRow` with the size of the given table's columns.
    pub fn new_for_table(table: &FirehoseTableSchema) -> Self {
        Row::new_with_width(table.columns.len())
    }

    /// Assigns values from another `Row` to this one.
    pub fn assign_from(
        &mut self,
        other: &Row,
    ) {
        if self.slots.len() != other.slots.len() {
            panic!(
                "Cannot assign from row with {} slots to row with {} slots",
                other.slots.len(),
                self.slots.len()
            );
        }
        for (i, value) in other.slots.iter().enumerate() {
            self.slots[i] = value.clone();
        }
    }

    /// Sets the value at the specified index.
    ///
    /// ## Arguments
    ///
    /// * `index`: The index of the value to set.
    /// * `value`: The value to set at the specified index, wrapped in an `Option<AnyArc>`.
    pub fn set_slot(
        &mut self,
        index: usize,
        value: Option<AnyArc>,
    ) {
        self.slots[index] = value;
    }

    /// Gets the value at the specified index, downcasting it to the specified type.
    ///
    /// ## Arguments
    /// * `index`: The index of the value to retrieve.
    pub fn get_slot<T: 'static>(
        &self,
        index: usize,
    ) -> Option<&T> {
        match self.slots.get(index)? {
            None => None,
            Some(value) => value.downcast_ref::<T>(),
        }
    }

    /// Gets the value at the specified index, downcasting it to the specified type.
    ///
    /// This method checks if the value exists and can be downcast to the specified type.
    ///
    /// ## Arguments
    ///
    /// * `index`: The index of the value to retrieve.
    ///
    /// ## Returns
    ///
    /// A `Result` that is `Ok(Some(&T))` if the value exists and can be downcast,
    /// `Ok(None)` if the value does not exist,
    /// and `Err(String)` if the value exists but cannot be downcast to the specified type.
    pub fn get_slot_checked<T: 'static>(
        &self,
        index: usize,
    ) -> Result<Option<&T>, String> {
        match self.get_slot_any_ref(index) {
            None => Ok(None),
            Some(value) => match value.downcast_ref::<T>() {
                Some(value) => Ok(Some(value)),
                None => Err(format!(
                    "Value at index {} cannot be downcast to type {}",
                    index,
                    std::any::type_name::<T>(),
                )),
            },
        }
    }

    /// Gets the untyped value at the specified index.
    ///
    /// ## Arguments
    ///
    /// * `index`: The index of the value to retrieve.
    ///
    /// ## Returns
    ///
    /// An `Option<&dyn Any>` representing the value at the specified index.
    pub fn get_slot_any_ref(
        &self,
        index: usize,
    ) -> Option<&dyn Any> {
        match self.slots.get(index)? {
            None => None,
            Some(value) => Some(value.as_ref()),
        }
    }

    /// Gets the `Option<Arc<dyn Any>>` wrapper for the untyped value at the specified index.
    pub fn get_slot_arc_any(
        &self,
        index: usize,
    ) -> Option<Arc<dyn Any>> {
        match self.slots.get(index)? {
            None => None,
            Some(value) => Some(value.clone()),
        }
    }

    /// Check if the row's format matches the schema of the table.
    fn fastcheck_schema(
        &self,
        schema: &FirehoseTableSchema,
    ) -> Result<(), String> {
        if self.slots.len() != schema.columns.len() {
            return Err(format!(
                "Row has {} slots, but table has {} columns",
                self.slots.len(),
                schema.columns.len()
            ));
        }
        Ok(())
    }

    /// Gets the value of a column by its name, downcasting it to the specified type.
    ///
    /// ## Arguments
    ///
    /// * `schema`: The schema of the table to which this row belongs.
    /// * `column_name`: The name of the column to retrieve the value from.
    ///
    /// ## Returns
    ///
    /// An `Option<&T>` where `T` is the type of the column value.
    pub fn get_column<T: 'static>(
        &self,
        schema: &FirehoseTableSchema,
        column_name: &str,
    ) -> Option<&T> {
        self.fastcheck_schema(schema).unwrap();

        let index = schema.check_column_index(column_name).unwrap();
        self.get_slot::<T>(index)
    }

    /// Gets the value of a column by its name, downcasting it to the specified type.
    ///
    /// This method checks if the value exists and can be downcast to the specified type.
    ///
    /// ## Arguments
    ///
    /// * `schema`: The schema of the table to which this row belongs.
    /// * `column_name`: The name of the column to retrieve the value from.
    ///
    /// ## Returns
    ///
    /// A `Result` that is `Ok(Some(&T))` if the value exists and can be downcast,
    /// `Ok(None)` if the value does not exist,
    /// and `Err(String)` if the value exists but cannot be downcast to the specified type.
    pub fn get_column_checked<T: 'static>(
        &self,
        schema: &FirehoseTableSchema,
        column_name: &str,
    ) -> Result<Option<&T>, String> {
        self.fastcheck_schema(schema).unwrap();

        let index = schema.check_column_index(column_name).unwrap();
        self.get_slot_checked::<T>(index)
    }

    /// Sets the value of a column by its name.
    ///
    /// ## Arguments
    ///
    /// * `schema`: The schema of the table to which this row belongs.
    /// * `column_name`: The name of the column to set the value for.
    /// * `value`: The value to set for the specified column, wrapped in an `Option<AnyArc>`.
    pub fn set_column(
        &mut self,
        schema: &FirehoseTableSchema,
        column_name: &str,
        value: Option<AnyArc>,
    ) {
        self.fastcheck_schema(schema).unwrap();

        let index = schema.check_column_index(column_name).unwrap();
        self.set_slot(index, value);
    }

    /// Sets multiple column values by their names.
    ///
    /// ## Arguments
    ///
    /// * `schema`: The schema of the table to which this row belongs.
    /// * `names`: An array of column names to set values for.
    /// * `values`: An array of values to set for the corresponding columns.
    pub fn set_columns<const K: usize>(
        &mut self,
        schema: &FirehoseTableSchema,
        names: &[&str; K],
        values: [AnyArc; K],
    ) {
        self.fastcheck_schema(schema).unwrap();

        let indices = schema.select_indices(names).unwrap();

        for (i, value) in values.into_iter().enumerate() {
            self.set_slot(indices[i], Some(value));
        }
    }

    /// Creates a new `BimmRow` with the specified column names and values.
    ///
    /// ## Arguments
    ///
    /// * `schema`: The schema of the table to which this row belongs.
    /// * `names`: An array of column names to set values for.
    /// * `values`: An array of values to set for the corresponding columns.
    ///
    /// ## Returns
    ///
    /// A new `BimmRow` instance with the specified values set for the given column names.
    pub fn new_with_columns<const K: usize>(
        schema: &FirehoseTableSchema,
        names: &[&str; K],
        values: [AnyArc; K],
    ) -> Self {
        let mut row = Row::new_for_table(schema);
        row.set_columns(schema, names, values);
        row
    }
}

/// Represents a batch of `BimmRow`, with `BimmTableSchema` as its schema.
#[derive(Debug, Clone)]
pub struct RowBatch {
    /// The schema of the table slice.
    pub schema: Arc<FirehoseTableSchema>,

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
    pub fn new<S>(
        schema: S,
        rows: Vec<Row>,
    ) -> Self
    where
        S: Into<Arc<FirehoseTableSchema>>,
    {
        let schema = schema.into();
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
    pub fn with_size<S>(
        schema: S,
        size: usize,
    ) -> Self
    where
        S: Into<Arc<FirehoseTableSchema>>,
    {
        let schema = schema.into();
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

    /// Assigns values from another `RowBatch` to this one.
    pub fn assign_from(
        &mut self,
        other: &RowBatch,
    ) {
        if self.schema != other.schema {
            panic!("Cannot assign from row batch with different schemas");
        }
        if self.rows.len() != other.rows.len() {
            panic!(
                "Cannot assign from row batch with {} rows to row batch with {} rows",
                other.rows.len(),
                self.rows.len()
            );
        }
        for (i, row) in other.rows.iter().enumerate() {
            self.rows[i].assign_from(row);
        }
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
    use crate::core::schema::{ColumnSchema, FirehoseTableSchema};
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::TensorData;
    use std::sync::Arc;

    #[test]
    fn test_get_value() {
        let schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
            ColumnSchema::new::<i32>("qux"),
        ]);

        let row = Row::new_with_columns(
            &schema,
            &["foo", "bar"],
            [Arc::new(42), Arc::new("Hello".to_string())],
        );

        assert_eq!(row.get_slot::<i32>(0), Some(&42));
        assert_eq!(row.get_slot::<String>(1), Some(&"Hello".to_string()));

        // Bad-type access should return None
        assert_eq!(row.get_slot::<String>(0), None);

        // Reading an empty column should return None
        assert_eq!(row.get_slot::<i32>(2), None);
        assert_eq!(row.get_column::<i32>(&schema, "qux"), None);

        assert_eq!(row.get_column::<i32>(&schema, "foo"), Some(&42));
        // Bad type access by name should return None
        assert_eq!(row.get_column::<String>(&schema, "foo"), None);
    }

    #[test]
    fn test_set_column() {
        let schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let mut row = Row::new_for_table(&schema);

        // Set values for columns
        row.set_column(&schema, "foo", Some(Arc::new(42)));
        row.set_column(&schema, "bar", Some(Arc::new("Hello".to_string())));

        assert_eq!(row.get_slot::<i32>(0), Some(&42));
        assert_eq!(row.get_slot::<String>(1), Some(&"Hello".to_string()));
    }

    #[test]
    fn test_get_checked() {
        let schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let row = Row::new_with_columns(&schema, &["foo"], [Arc::new(42)]);

        // Valid type access
        assert_eq!(
            row.get_column_checked::<i32>(&schema, "foo").unwrap(),
            Some(&42)
        );

        // Invalid type access
        assert!(row.get_column_checked::<f64>(&schema, "foo").is_err());

        // Accessing an empty column
        assert_eq!(row.get_column_checked::<i32>(&schema, "bar"), Ok(None));
    }

    #[test]
    #[should_panic(expected = "Row has 2 slots, but table has 3 columns")]
    fn test_row_with_invalid_column_count() {
        let schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
            ColumnSchema::new::<f64>("baz"),
        ]);

        let row = Row::new_with_width(2);

        row.get_column::<i32>(&schema, "foo");
    }

    #[test]
    fn test_row_with_basic_types() {
        let schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let row = Row::new_with_columns(
            &schema,
            &["foo", "bar"],
            [Arc::new(42), Arc::new("Hello".to_string())],
        );

        assert_eq!(row.get_slot::<i32>(0), Some(&42));
        assert_eq!(row.get_slot::<String>(1), Some(&"Hello".to_string()));
    }

    #[test]
    fn test_row_with_tensor() {
        let schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
            ColumnSchema::new::<Tensor<NdArray, 2>>("qux"),
        ]);

        let device = Default::default();

        let row = Row::new_with_columns(
            &schema,
            &["foo", "bar", "qux"],
            [
                Arc::new(42),
                Arc::new("World".to_string()),
                Arc::new(Tensor::<NdArray, 2>::from_data(
                    [[1.0, 2.0], [3.0, 4.0]],
                    &device,
                )),
            ],
        );

        assert_eq!(row.get_slot::<i32>(0), Some(&42));
        assert_eq!(row.get_slot::<String>(1), Some(&"World".to_string()));

        let tensor = row.get_slot::<Tensor<NdArray, 2>>(2).unwrap();
        tensor
            .clone()
            .to_data()
            .assert_eq(&TensorData::from([[1.0, 2.0], [3.0, 4.0]]), false);
    }

    #[test]
    fn test_with_size() {
        let schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]);

        let batch = RowBatch::with_size(Arc::new(schema), 3);
        assert_eq!(batch.len(), 3);
        assert!(!batch.is_empty());
    }

    #[test]
    fn test_row_batch_with_basic_types() {
        let schema = FirehoseTableSchema::from_columns(&[
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
        let schema = FirehoseTableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);

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
        let schema = FirehoseTableSchema::from_columns(&[
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
