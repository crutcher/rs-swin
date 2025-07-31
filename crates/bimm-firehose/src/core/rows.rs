use crate::core::ValueBox;
use crate::core::schema::FirehoseTableSchema;
use anyhow::bail;
use std::any::Any;
use std::fmt::Debug;
use std::ops::{Index, IndexMut, Range};
use std::sync::Arc;

/// Represents a row in a Firehose table, containing values for each column.
pub struct ValueRow {
    schema: Arc<FirehoseTableSchema>,
    slots: Vec<Option<ValueBox>>,
}

impl Debug for ValueRow {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "ValueRow ")?;
        if f.alternate() {
            self.inner_fmt_vert(f)
        } else {
            self.inner_fmt_horiz(f)
        }
    }
}

impl ValueRow {
    /// Vertical inner formatting function for `ValueRow`.
    pub(crate) fn inner_fmt_vert(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{{\n")?;
        for (idx, (name, value)) in self.iter().enumerate() {
            if idx > 0 {
                write!(f, "\n")?;
            }

            let data_type = &self.schema[name].data_type;

            write!(f, "  # {}\n", data_type.type_name)?;
            write!(f, "  {}: ", name)?;
            match value {
                Some(v) => write!(f, "{:?},\n", v),
                None => write!(f, "None,\n"),
            }?;
        }
        write!(f, "}}")
    }

    /// Horizontal inner formatting function for `ValueRow`.
    pub(crate) fn inner_fmt_horiz(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{{ ")?;
        for (idx, (name, value)) in self.iter().enumerate() {
            if idx > 0 {
                write!(f, ", ")?;
            }
            write!(f, "{}: ", name)?;
            match value {
                Some(v) => write!(f, "{:?}", v),
                None => write!(f, "None"),
            }?;
        }
        write!(f, " }}")
    }
}

impl ValueRow {
    /// Creates a new `ValueRow` with the given schema and initializes all slots to `None`.
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        let mut slots = Vec::with_capacity(schema.columns.len());
        slots.resize_with(schema.columns.len(), || None);
        ValueRow { schema, slots }
    }

    /// Returns the schema of the row.
    pub fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    /// Returns an iterator over the column names and their corresponding values.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &Option<ValueBox>)> {
        self.schema
            .columns
            .iter()
            .map(|c| c.name.as_str())
            .zip(self.slots.iter())
    }

    /// Returns true if the row has a value for the specified column name.
    ///
    /// # Arguments
    ///
    /// - `column_name`: The name of the column to check for a value.
    ///
    /// # Panics
    ///
    /// Panics if the column name does not exist in the schema.
    ///
    /// # Returns
    ///
    /// A boolean indicating whether the row has a value for the specified column.
    pub fn has_column_value(
        &self,
        column_name: &str,
    ) -> bool {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index].is_some()
    }

    /// Gets a reference to the value of the specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name`: The name of the column to retrieve the value from.
    ///
    /// # Returns
    ///
    /// An `Option<&ValueBox>` containing a reference to the value of the specified column,
    /// or `None` if the column does not exist or has no value.
    pub fn get(
        &self,
        column_name: &str,
    ) -> Option<&ValueBox> {
        let index = self.schema.column_index(column_name)?;
        self.slots.get(index)?.as_ref()
    }

    /// Sets the value of the specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name`: The name of the column to set the value for.
    /// * `value`: The value to set for the specified column, wrapped in a `ValueBox`.
    ///
    /// # Panics
    ///
    /// Panics if the column name does not exist in the schema.
    pub fn set(
        &mut self,
        column_name: &str,
        value: ValueBox,
    ) {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index] = Some(value);
    }

    /// Clear the value of the specified column to `None`.
    pub fn clear(
        &mut self,
        column_name: &str,
    ) {
        let index = self.schema.check_column_index(column_name).unwrap();
        self.slots[index] = None;
    }
}

impl Index<&str> for ValueRow {
    type Output = ValueBox;

    /// Returns a reference to the value of the specified column.
    ///
    /// # Arguments
    ///
    /// * `column_name`: The name of the column to retrieve the value from.
    ///
    /// # Panics
    ///
    /// Panics if the column name does not exist in the schema or if the value is not set.
    ///
    /// # Returns
    ///
    /// A reference to the value of the specified column.
    fn index(
        &self,
        column_name: &str,
    ) -> &Self::Output {
        let index = self
            .schema
            .check_column_index(column_name)
            .expect("Column not found in schema");

        match self.slots[index].as_ref() {
            Some(value) => value,
            None => panic!("Column {} has no value", column_name),
        }
    }
}

pub struct ValueRowBatch {
    /// The schema of the row batch.
    schema: Arc<FirehoseTableSchema>,

    /// The rows in the batch.
    rows: Vec<ValueRow>,
}

impl Debug for ValueRowBatch {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "ValueRowBatch {{\n")?;
        for (idx, row) in self.iter().enumerate() {
            write!(f, "  {idx}: ")?;
            row.inner_fmt_horiz(f)?;
            write!(f, ",\n")?;
        }
        write!(f, "}}")
    }
}

impl ValueRowBatch {
    /// Creates a new `ValueRowBatch` with the given schema and initializes an empty vector of rows.
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        ValueRowBatch {
            schema,
            rows: Vec::new(),
        }
    }

    /// The number of rows in the batch.
    pub fn len(&self) -> usize {
        self.rows.len()
    }

    /// Checks if the batch is empty.
    pub fn is_empty(&self) -> bool {
        self.rows.is_empty()
    }

    /// Returns an iterator over the rows in the batch.
    pub fn iter(&self) -> impl Iterator<Item = &ValueRow> {
        self.rows.iter()
    }

    /// Returns a mutable iterator over the rows in the batch.
    pub fn iter_mut(&mut self) -> impl Iterator<Item = &mut ValueRow> {
        self.rows.iter_mut()
    }

    /// Adds a new row to the batch.
    ///
    /// # Arguments
    ///
    /// * `row`: The `ValueRow` to add to the batch.
    ///
    /// # Panics
    ///
    /// Panics if the row's schema does not match the batch's schema.
    pub fn add_row(
        &mut self,
        row: ValueRow,
    ) {
        if row.schema != self.schema {
            panic!("Cannot add row with different schema");
        }
        self.rows.push(row);
    }

    pub fn new_row(&mut self) -> &mut ValueRow {
        let row = ValueRow::new(self.schema.clone());
        self.rows.push(row);
        self.rows.last_mut().unwrap()
    }
}

impl Index<usize> for ValueRowBatch {
    type Output = ValueRow;

    /// Returns a reference to the row at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        &self.rows[index]
    }
}

impl IndexMut<usize> for ValueRowBatch {
    /// Returns a mutable reference to the row at the specified index.
    ///
    /// # Arguments
    ///
    /// * `index`: The index of the row to retrieve.
    fn index_mut(
        &mut self,
        index: usize,
    ) -> &mut Self::Output {
        &mut self.rows[index]
    }
}

impl Index<Range<usize>> for ValueRowBatch {
    type Output = [ValueRow];

    /// Returns a slice of rows for the specified range.
    ///
    /// # Arguments
    ///
    /// * `range`: The range of indices to retrieve.
    fn index(
        &self,
        range: Range<usize>,
    ) -> &Self::Output {
        &self.rows[range]
    }
}

impl IndexMut<Range<usize>> for ValueRowBatch {
    /// Returns a mutable slice of rows for the specified range.
    ///
    /// # Arguments
    ///
    /// * `range`: The range of indices to retrieve.
    fn index_mut(
        &mut self,
        range: Range<usize>,
    ) -> &mut Self::Output {
        &mut self.rows[range]
    }
}

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
    ) -> anyhow::Result<Option<&T>> {
        match self.get_slot_any_ref(index) {
            None => Ok(None),
            Some(value) => match value.downcast_ref::<T>() {
                Some(value) => Ok(Some(value)),
                None => bail!(
                    "Value at index {} cannot be downcast to type {}",
                    index,
                    std::any::type_name::<T>(),
                ),
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
    ) -> anyhow::Result<()> {
        if self.slots.len() != schema.columns.len() {
            bail!(
                "Row has {} slots, but table has {} columns",
                self.slots.len(),
                schema.columns.len()
            );
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
    ) -> anyhow::Result<Option<&T>> {
        self.fastcheck_schema(schema)?;
        let index = schema.check_column_index(column_name)?;
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
        join_fn: impl Fn(&Vec<Option<&T>>) -> anyhow::Result<V>,
    ) -> anyhow::Result<V> {
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
        assert_eq!(row.get_column_checked::<i32>(&schema, "bar").unwrap(), None);
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
            batch
                .join_column("foo", |vs| {
                    let x: i32 = vs.iter().filter_map(|&v| v).sum();
                    Ok(x)
                })
                .unwrap(),
            142_i32,
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
            batch
                .join_column("bar", |vs: &Vec<Option<&String>>| {
                    let joined = vs
                        .iter()
                        .filter_map(|&v| v)
                        .cloned()
                        .collect::<Vec<_>>()
                        .join(", ");
                    Ok(joined)
                })
                .unwrap(),
            "Hello, World",
        );
    }

    /// Ensures that `ValueRow` is `Send`, allowing it to be safely shared across threads.
    #[allow(dead_code)]
    const VALUE_ROW_IS_SEND: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<ValueRow>();
    };

    #[test]
    fn test_value_row_creation() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let row = ValueRow::new(schema.clone());
        assert_eq!(row.slots.len(), 2);
        assert_eq!(row.has_column_value("foo"), false);
        assert_eq!(row.has_column_value("bar"), false);
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct MyStruct {
        pub value: i32,
    }

    #[test]
    fn test_row_mutation() -> anyhow::Result<()> {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<MyStruct>("bar"),
            ColumnSchema::new::<String>("qux"),
        ]));

        let mut row = ValueRow::new(schema.clone());
        assert!(!row.has_column_value("foo"));
        assert!(!row.has_column_value("bar"));

        row.set("foo", ValueBox::serializing(42)?);
        assert!(row.has_column_value("foo"));
        assert_eq!(row["foo"].deserializing::<f32>()?, 42_f32);

        let my_struct = MyStruct { value: 100 };
        row.set("bar", ValueBox::boxing(my_struct.clone()));
        assert!(row.has_column_value("bar"));

        assert_eq!(
            format!("{:?}", row),
            "ValueRow { foo: {42}, bar: [Any { .. }], qux: None }"
        );
        assert_eq!(
            format!("{:#?}", row),
            indoc::indoc! {r#"
              ValueRow {
                # i32
                foo: {42},

                # bimm_firehose::core::rows::tests::MyStruct
                bar: [Any { .. }],

                # alloc::string::String
                qux: None,
              }"#},
        );

        Ok(())
    }

    #[test]
    fn test_value_row_batch_debug() {
        let schema = Arc::new(FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("bar"),
        ]));

        let mut batch = ValueRowBatch::new(schema.clone());
        let row = batch.new_row();
        row.set("foo", ValueBox::serializing(42).unwrap());

        let row = batch.new_row();
        row.set("bar", ValueBox::serializing("Hello").unwrap());

        assert_eq!(
            format!("{:#?}", batch),
            indoc::indoc! {r#"
                ValueRowBatch {
                  0: { foo: {42}, bar: None },
                  1: { foo: None, bar: {"Hello"} },
                }"#}
        );
    }
}
