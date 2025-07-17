use crate::data::table::schema::BimmTableSchema;
use std::sync::Arc;

/// Represents a boxed value that can hold any type.
pub type AnyArc = Arc<dyn std::any::Any>;

/// Represents a row in a Bimm table, containing values for each column.
#[derive(Debug, Clone)]
pub struct BimmRow {
    /// The values in the row, where each value is an `Option<AnyArc>`.
    pub slots: Vec<Option<AnyArc>>,
}

impl BimmRow {
    /// Creates a new `BurnRow` with the given values.
    pub fn new_with_width(size: usize) -> Self {
        let mut slots = Vec::with_capacity(size);
        slots.resize_with(size, || None);

        BimmRow { slots }
    }

    /// Creates an empty `BurnRow` with the size of the given table's columns.
    pub fn new_for_table(table: &BimmTableSchema) -> Self {
        BimmRow::new_with_width(table.columns.len())
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

    /// Gets the untyped value at the specified index.
    ///
    /// ## Arguments
    ///
    /// * `index`: The index of the value to retrieve.
    ///
    /// ## Returns
    ///
    /// An `Option<&dyn std::any::Any>` representing the value at the specified index.
    pub fn get_untyped_slot(
        &self,
        index: usize,
    ) -> Option<&dyn std::any::Any> {
        match self.slots.get(index)? {
            None => None,
            Some(value) => Some(value.as_ref()),
        }
    }

    /// Check if the row's format matches the schema of the table.
    fn fastcheck_schema(
        &self,
        schema: &BimmTableSchema,
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
        schema: &BimmTableSchema,
        column_name: &str,
    ) -> Option<&T> {
        self.fastcheck_schema(schema).unwrap();

        let index = schema.check_column_index(column_name).unwrap();
        self.get_slot::<T>(index)
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
        schema: &BimmTableSchema,
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
        schema: &BimmTableSchema,
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
        schema: &BimmTableSchema,
        names: &[&str; K],
        values: [AnyArc; K],
    ) -> Self {
        let mut row = BimmRow::new_for_table(schema);
        row.set_columns(schema, names, values);
        row
    }
}

#[cfg(test)]
mod tests {
    use crate::data::table::rows::BimmRow;
    use crate::data::table::schema::{BimmColumnSchema, BimmTableSchema};
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::TensorData;
    use std::sync::Arc;

    #[test]
    fn test_get_value() {
        let schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("foo"),
            BimmColumnSchema::new::<String>("bar"),
            BimmColumnSchema::new::<i32>("qux"),
        ]);

        let row = BimmRow::new_with_columns(
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
        let schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("foo"),
            BimmColumnSchema::new::<String>("bar"),
        ]);

        let mut row = BimmRow::new_for_table(&schema);

        // Set values for columns
        row.set_column(&schema, "foo", Some(Arc::new(42)));
        row.set_column(&schema, "bar", Some(Arc::new("Hello".to_string())));

        assert_eq!(row.get_slot::<i32>(0), Some(&42));
        assert_eq!(row.get_slot::<String>(1), Some(&"Hello".to_string()));
    }

    #[test]
    #[should_panic(expected = "Row has 2 slots, but table has 3 columns")]
    fn test_row_with_invalid_column_count() {
        let schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("foo"),
            BimmColumnSchema::new::<String>("bar"),
            BimmColumnSchema::new::<f64>("baz"),
        ]);

        let row = BimmRow::new_with_width(2);

        row.get_column::<i32>(&schema, "foo");
    }

    #[test]
    fn test_row_with_basic_types() {
        let schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("foo"),
            BimmColumnSchema::new::<String>("bar"),
        ]);

        let row = BimmRow::new_with_columns(
            &schema,
            &["foo", "bar"],
            [Arc::new(42), Arc::new("Hello".to_string())],
        );

        assert_eq!(row.get_slot::<i32>(0), Some(&42));
        assert_eq!(row.get_slot::<String>(1), Some(&"Hello".to_string()));
    }

    #[test]
    fn test_row_with_tensor() {
        let schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("foo"),
            BimmColumnSchema::new::<String>("bar"),
            BimmColumnSchema::new::<Tensor<NdArray, 2>>("qux"),
        ]);

        let device = Default::default();

        let row = BimmRow::new_with_columns(
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
}
