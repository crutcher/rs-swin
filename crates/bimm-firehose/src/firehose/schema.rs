use crate::firehose::identifiers;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::{Index, IndexMut};

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

/// Namespace and name of a column operator.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperatorId {
    /// The namespace of the operator.
    pub namespace: String,

    /// The name of the operator.
    pub name: String,
}

impl OperatorId {
    /// Creates a new `OperatorId` with the given namespace and name.
    pub fn new(
        namespace: &str,
        name: &str,
    ) -> Self {
        identifiers::check_ident(namespace).expect("Invalid namespace");
        identifiers::check_ident(name).expect("Invalid name");

        OperatorId {
            namespace: namespace.to_string(),
            name: name.to_string(),
        }
    }
}

/// A specification for a column operator, including its ID, description, and configuration.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BuildOperatorSpec {
    /// The ID of the operator.
    pub id: OperatorId,

    /// The description of the operator.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// Additional configuration for the operation, serialized as JSON.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    #[serde(default)]
    pub config: serde_json::Value,
}

/// A build plan for columns in a table schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnBuildPlan {
    /// The spec for the operator that will be used to build the columns.
    pub operator: BuildOperatorSpec,

    /// The input column bindings ``{parameter_name: column_name}``.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub inputs: BTreeMap<String, String>,

    /// The output column bindings ``{parameter_name: column_name}``.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub outputs: BTreeMap<String, String>,
}

/// A description of a column in a data table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ColumnSchema {
    /// The name of the column.
    pub name: String,

    /// An optional description of the column.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// The type of the column.
    pub data_type: DataTypeDescription,
}

impl ColumnSchema {
    /// Creates a new `DataColumnDescription` with the given name and type.
    pub fn new<T>(name: &str) -> Self {
        identifiers::check_ident(name).unwrap();
        ColumnSchema {
            name: name.to_string(),
            description: None,
            data_type: DataTypeDescription::new::<T>(),
        }
    }

    /// Extends the schema with a description.
    ///
    /// ## Arguments
    ///
    /// - `description`: The description to attach to the column.
    ///
    /// ## Returns
    ///
    /// A new `BimmColumnSchema` with the description attached.
    pub fn with_description(
        self,
        description: &str,
    ) -> Self {
        ColumnSchema {
            description: Some(description.to_string()),
            ..self
        }
    }
}

/// Bimm Table Schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct TableSchema {
    /// The columns in the table.
    pub columns: Vec<ColumnSchema>,

    /// Build plans for the table.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    #[serde(default)]
    pub build_plans: Vec<ColumnBuildPlan>,
}

impl Index<usize> for TableSchema {
    type Output = ColumnSchema;

    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        &self.columns[index]
    }
}

impl IndexMut<usize> for TableSchema {
    fn index_mut(
        &mut self,
        index: usize,
    ) -> &mut Self::Output {
        &mut self.columns[index]
    }
}

impl Index<&str> for TableSchema {
    type Output = ColumnSchema;

    fn index(
        &self,
        index: &str,
    ) -> &Self::Output {
        let name = index;
        self.column_index(name)
            .and_then(|idx| self.columns.get(idx))
            .expect("Column not found")
    }
}

impl IndexMut<&str> for TableSchema {
    fn index_mut(
        &mut self,
        index: &str,
    ) -> &mut Self::Output {
        let name = index;
        let idx = self.check_column_index(name).expect("Column not found");
        &mut self.columns[idx]
    }
}

/// Topological build order for columns in a table schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize, Default)]
pub struct ColumnBuildOrder {
    /// Non-buildable columns (those without build info).
    pub static_columns: Vec<String>,

    /// The order in which columns can be built.
    pub topo_order: Vec<String>,
}

impl TableSchema {
    /// Creates a new `DataTableDescription` with the given columns.
    #[must_use]
    pub fn from_columns(columns: &[ColumnSchema]) -> Self {
        let mut schema = Self {
            columns: Vec::new(),
            build_plans: Vec::new(),
        };

        for column in columns {
            schema.add_column(column.clone());
        }

        schema
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
            Err(format!("Duplicate column name '{name}'"))
        } else {
            Ok(())
        }
    }

    /// Adds a column to the table description.
    pub fn add_column(
        &mut self,
        column: ColumnSchema,
    ) {
        self.check_name(&column.name).unwrap();

        self.columns.push(column);
    }

    /// Adds a build plan to the table description.
    pub fn add_build_plan(
        &mut self,
        plan: ColumnBuildPlan,
    ) -> Result<(), String> {
        // Check that all the input and output columns exist.
        for cname in plan.inputs.values() {
            if self.column_index(cname).is_none() {
                return Err(format!(
                    "Input column '{cname}' does not exist in the schema"
                ));
            }
        }
        for cname in plan.outputs.values() {
            if self.column_index(cname).is_none() {
                return Err(format!(
                    "Output column '{cname}' does not exist in the schema"
                ));
            }

            for alt_plan in &self.build_plans {
                if alt_plan.outputs.values().any(|v| v == cname) {
                    return Err(format!(
                        "Output column '{cname}' already exists in another build plan"
                    ));
                }
            }
        }

        self.build_plans.push(plan);
        Ok(())
    }

    /// Renames a column in the table description.
    pub fn rename_column(
        &mut self,
        old_name: &str,
        new_name: &str,
    ) -> Result<(), String> {
        let index = match self.column_index(old_name) {
            Some(idx) => idx,
            None => {
                return Err(format!("Column '{old_name}' not found"));
            }
        };

        if old_name == new_name {
            // No change needed
            return Ok(());
        }

        if self.column_index(new_name).is_some() {
            return Err(format!("Column name '{new_name}' already exists"));
        }

        self.check_name(new_name)?;

        // Update the column name in the schema.
        self.columns[index].name = new_name.to_string();

        // Update all references to the old column name in build plans.
        for plan in &mut self.build_plans {
            for cname in plan.inputs.values_mut() {
                if cname == old_name {
                    *cname = new_name.to_string();
                }
            }
            for cname in plan.outputs.values_mut() {
                if cname == old_name {
                    *cname = new_name.to_string();
                }
            }
        }

        Ok(())
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
    /// use bimm_firehose::firehose::{TableSchema, ColumnSchema};
    /// let table = TableSchema::from_columns(&[
    ///     ColumnSchema::new::<i32>("foo"),
    ///     ColumnSchema::new::<String>("bar"),
    ///     ColumnSchema::new::<Option<usize>>("baz"),
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
    use super::*;
    use indoc::indoc;

    #[test]
    fn test_data_type_description() {
        let schema = DataTypeDescription::new::<Option<i32>>();
        assert_eq!(schema.type_name, std::any::type_name::<Option<i32>>(),);
    }

    #[test]
    fn test_column_description() {
        let column = ColumnSchema::new::<Option<i32>>("abc");

        assert_eq!(column.name, "abc");
        assert_eq!(column.data_type, DataTypeDescription::new::<Option<i32>>());
    }

    #[test]
    fn test_schema() {
        let mut schema = TableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);
        schema.add_column(ColumnSchema::new::<String>("bar"));

        // Index<usize>
        assert_eq!(schema[0].name, "foo");
        // IndexMut<usize>
        schema[0].description = Some("Overwritten".to_string());

        // Index<&str>
        assert_eq!(schema["foo"].name, "foo");
        // IndexMut<usize>
        schema["foo"].description = Some("An integer column".to_string());

        assert_eq!(schema.columns.len(), 2);
        assert_eq!(schema.columns[0].name, "foo");
        assert_eq!(schema.columns[1].name, "bar");

        assert_eq!(schema.column_index("foo"), Some(0));
        assert_eq!(schema.column_index("bar"), Some(1));
        assert_eq!(schema.column_index("qux"), None);

        assert_eq!(
            serde_json::to_string_pretty(&schema).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "foo",
                      "description": "An integer column",
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

        // No-op.
        schema.rename_column("foo", "foo").unwrap();

        schema.rename_column("foo", "xxx").unwrap();
        assert_eq!(
            serde_json::to_string_pretty(&schema).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "xxx",
                      "description": "An integer column",
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

    #[test]
    #[should_panic(expected = "Duplicate column name 'foo'")]
    fn conflicting_column_names_on_validate() {
        let _schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("foo"),
            ColumnSchema::new::<String>("foo"),
        ]);
    }

    #[test]
    #[should_panic(expected = "Duplicate column name 'foo'")]
    fn conflicting_column_names_on_add() {
        let mut schema = TableSchema::from_columns(&[ColumnSchema::new::<i32>("foo")]);
        schema.add_column(ColumnSchema::new::<String>("foo"));
    }
}
