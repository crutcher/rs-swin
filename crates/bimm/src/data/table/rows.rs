use crate::data::table::schema::BimmTableSchema;

/// Represents a boxed value that can hold any type.
pub type AnyBox = Box<dyn std::any::Any>;

/// Represents a row in a Bimm table, containing values for each column.
#[derive(Debug)]
pub struct BimmRow {
    /// The values in the row, where each value is an `Option<AnyBox>`.
    pub values: Vec<Option<AnyBox>>,
}

impl BimmRow {
    /// Creates a new `BurnRow` with the given values.
    pub fn new_with_width(size: usize) -> Self {
        let mut values = Vec::with_capacity(size);
        values.resize_with(size, || None);

        BimmRow { values }
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
    /// * `value`: The value to set at the specified index, wrapped in an `Option<AnyBox>`.
    pub fn set_value(
        &mut self,
        index: usize,
        value: Option<AnyBox>,
    ) {
        self.values[index] = value;
    }

    /// Gets the value at the specified index, downcasting it to the specified type.
    ///
    /// ## Arguments
    /// * `index`: The index of the value to retrieve.
    pub fn get_value<T: 'static>(
        &self,
        index: usize,
    ) -> Option<&T> {
        match self.values.get(index)? {
            None => None,
            Some(value) => value.downcast_ref::<T>(),
        }
    }

    fn fastcheck_schema(
        &self,
        schema: &BimmTableSchema,
    ) -> Result<(), String> {
        if self.values.len() != schema.columns.len() {
            return Err(format!(
                "Row has {} values, but table has {} columns",
                self.values.len(),
                schema.columns.len()
            ));
        }
        Ok(())
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
        values: [AnyBox; K],
    ) {
        self.fastcheck_schema(schema).unwrap();

        let indices = schema.select_indices(names).unwrap();

        for (i, value) in values.into_iter().enumerate() {
            self.set_value(indices[i], Some(value));
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
        values: [AnyBox; K],
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
                Box::new(42),
                Box::new("World".to_string()),
                Box::new(Tensor::<NdArray, 2>::from_data(
                    [[1.0, 2.0], [3.0, 4.0]],
                    &device,
                )),
            ],
        );

        assert_eq!(row.get_value::<i32>(0), Some(&42));
        assert_eq!(row.get_value::<String>(1), Some(&"World".to_string()));

        let tensor = row.get_value::<Tensor<NdArray, 2>>(2).unwrap();
        tensor
            .clone()
            .to_data()
            .assert_eq(&TensorData::from([[1.0, 2.0], [3.0, 4.0]]), false);
    }
}
