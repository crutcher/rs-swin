use crate::data::table::identifiers;
use serde::{Deserialize, Serialize};

/// A serializable description of a data type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataTypeDescription {
    /// The name of the data type.
    pub type_name: String,
}

impl DataTypeDescription {
    /// Creates a `DataTypeDescription` from a type name.
    pub fn from_type_name(type_name: &str) -> Self {
        DataTypeDescription {
            type_name: type_name.to_string(),
        }
    }

    /// Creates a `DataTypeDescription` for a specific type `T`.
    ///
    /// This function uses `std::any::type_name` to get the type name of `T`.
    pub fn new<T>() -> Self {
        Self::from_type_name(std::any::type_name::<T>())
    }
}

/// A description of a column in a data table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BimmColumnSchema {
    /// The name of the column.
    pub name: String,

    /// The type of the column.
    pub data_type: DataTypeDescription,
}

impl BimmColumnSchema {
    /// Creates a new `DataColumnDescription` with the given name and type.
    pub fn new<T>(name: &str) -> Self {
        identifiers::check_ident(name).unwrap();
        BimmColumnSchema {
            name: name.to_string(),
            data_type: DataTypeDescription::new::<T>(),
        }
    }
}

/// Bimm Table Schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BimmTableSchema {
    /// The columns in the table.
    pub columns: Vec<BimmColumnSchema>,
}

impl BimmTableSchema {
    /// Creates a new `DataTableDescription` with the given columns.
    pub fn from_columns(columns: &[BimmColumnSchema]) -> Self {
        let columns = columns.to_vec();
        let table = Self { columns };

        table.validate().unwrap();

        table
    }

    /// Validates the table description.
    pub fn validate(&self) -> Result<(), String> {
        let mut seen_names = std::collections::HashSet::new();
        for column in &self.columns {
            if !seen_names.insert(column.name.clone()) {
                return Err(format!("Duplicate column name: '{}'", column.name));
            }
            if column.data_type.type_name.is_empty() {
                return Err(format!("Column '{}' has an empty type name", column.name));
            }
        }
        Ok(())
    }

    /// Checks if a column name is invalid, or conflicts with existing columns.
    ///
    /// ## Arguments
    ///
    /// - `name`: The name of the column to check.
    fn check_name(
        &self,
        name: &str,
    ) -> Result<(), String> {
        identifiers::check_ident(name).unwrap();

        if self.columns.iter().any(|c| c.name == name) {
            Err(format!("Column name '{name}' already exists"))
        } else {
            Ok(())
        }
    }

    /// Adds a column to the table description.
    pub fn add_column(
        &mut self,
        column: BimmColumnSchema,
    ) {
        self.check_name(&column.name).unwrap();
        self.columns.push(column);
    }

    /// Returns the index of a column by its name.
    ///
    /// ## Arguments
    ///
    /// - `name`: The name of the column to find.
    ///
    /// ## Returns
    ///
    /// An `Option<usize>` containing the index of the column if found, or `None` if not found.
    pub fn column_index(
        &self,
        name: &str,
    ) -> Option<usize> {
        for (idx, column) in self.columns.iter().enumerate() {
            if column.name == name {
                return Some(idx);
            }
        }
        None
    }

    /// Checks if a column with the given name exists and returns its index.
    ///
    /// ## Arguments
    ///
    /// - `name`: The name of the column to check.
    ///
    /// ## Returns
    ///
    /// A `Result<usize, String>` where:
    /// - `Ok(usize)` is the index of the column if it exists.
    /// - `Err(String)` is an error message if the column does not exist.
    pub fn check_column_index(
        &self,
        name: &str,
    ) -> Result<usize, String> {
        self.column_index(name)
            .ok_or_else(|| format!("Column '{name}' not found"))
    }

    /// Selects the indices of the columns with the given names.
    ///
    /// ## Arguments
    ///
    /// - `names`: An array of column names to select.
    ///
    /// ## Returns
    ///
    /// A `Result<[usize; K], String>` where:
    /// - `Ok([usize; K])` is an array of indices corresponding to the column names.
    /// - `Err(String)` is an error message if any of the column names do not exist.
    ///
    /// ## Example
    ///
    /// ```rust
    /// use bimm::data::table::schema::{BimmTableSchema, BimmColumnSchema};
    /// let table = BimmTableSchema::from_columns(&[
    ///     BimmColumnSchema::new::<i32>("foo"),
    ///     BimmColumnSchema::new::<String>("bar"),
    ///     BimmColumnSchema::new::<Option<usize>>("baz"),
    /// ]);
    ///
    /// let [foo, baz] = table.select_indices(&["foo", "baz"]).unwrap();
    /// assert_eq!(foo, 0);
    /// assert_eq!(baz, 2);
    /// ```
    pub fn select_indices<const K: usize>(
        &self,
        names: &[&str; K],
    ) -> Result<[usize; K], String> {
        let mut indices = [0; K];
        for (i, name) in names.iter().enumerate() {
            indices[i] = self.check_column_index(name)?
        }
        Ok(indices)
    }
}

#[cfg(test)]
mod tests {
    use crate::data::table::schema::{BimmColumnSchema, BimmTableSchema, DataTypeDescription};
    use indoc::indoc;

    #[test]
    fn test_data_type_description() {
        let type_desc = DataTypeDescription::new::<Option<i32>>();
        assert_eq!(type_desc.type_name, std::any::type_name::<Option<i32>>(),);
    }

    #[test]
    fn test_column_description() {
        let column = BimmColumnSchema::new::<Option<i32>>("abc");

        assert_eq!(column.name, "abc");
        assert_eq!(column.data_type, DataTypeDescription::new::<Option<i32>>());
    }

    #[test]
    fn test_data_table_description() {
        let table_desc = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("foo"),
            BimmColumnSchema::new::<String>("bar"),
        ]);

        assert_eq!(table_desc.columns.len(), 2);
        assert_eq!(table_desc.columns[0].name, "foo");
        assert_eq!(table_desc.columns[1].name, "bar");

        assert_eq!(
            serde_json::to_string_pretty(&table_desc).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "foo",
                      "data_type": {
                        "type_name": "i32"
                      }
                    },
                    {
                      "name": "bar",
                      "data_type": {
                        "type_name": "alloc::string::String"
                      }
                    }
                  ]
                }"#
            }
        );
    }
}
