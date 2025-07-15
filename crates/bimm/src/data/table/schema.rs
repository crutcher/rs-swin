use crate::data::table::identifiers;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;
use std::ops::{Index, IndexMut};

/// A serializable description of a data type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BimmDataTypeDescription {
    /// The name of the data type.
    pub type_name: String,
}

impl BimmDataTypeDescription {
    /// Creates a `DataTypeDescription` from a type name.
    pub fn from_type_name(type_name: &str) -> Self {
        BimmDataTypeDescription {
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

/// Column build information.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BimmColumnBuildInfo {
    /// The name of the operation that builds this column.
    pub op: String,

    /// The bindings for the operation, serialized as a map of name-value pairs.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub deps: BTreeMap<String, String>,

    /// Additional arguments for the operation, serialized as JSON.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub params: BTreeMap<String, serde_json::Value>,
}

/// A description of a column in a data table.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BimmColumnSchema {
    /// The name of the column.
    pub name: String,

    /// An optional description of the column.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// The type of the column.
    pub data_type: BimmDataTypeDescription,

    /// Build information for the column, if applicable.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub build_info: Option<BimmColumnBuildInfo>,
}

impl BimmColumnSchema {
    /// Creates a new `DataColumnDescription` with the given name and type.
    pub fn new<T>(name: &str) -> Self {
        identifiers::check_ident(name).unwrap();
        BimmColumnSchema {
            name: name.to_string(),
            description: None,
            data_type: BimmDataTypeDescription::new::<T>(),
            build_info: None,
        }
    }

    /// Sets the description of the column.
    pub fn with_description(
        mut self,
        description: &str,
    ) -> Self {
        self.description = Some(description.to_string());
        self
    }

    /// Attaches build information to the column.
    ///
    /// ## Arguments
    ///
    /// - `op_name`: The name of the operation that builds this column.
    /// - `deps`: A vector of column names that this column depends on.
    /// - `params`: A vector of additional parameters for the operation, serialized as JSON.
    pub fn with_build_info(
        mut self,
        op_name: &str,
        deps: &[(&str, &str)],
        params: &[(&str, serde_json::Value)],
    ) -> Self {
        let deps = deps
            .iter()
            .map(|(pname, cname)| (pname.to_string(), cname.to_string()))
            .collect::<BTreeMap<_, _>>();

        let params = params
            .iter()
            .map(|(pname, value)| (pname.to_string(), value.clone()))
            .collect::<BTreeMap<_, _>>();

        self.build_info = Some(BimmColumnBuildInfo {
            op: op_name.to_string(),
            deps,
            params,
        });

        self
    }

    /// Computes a topological build order for the columns based on their dependencies.
    ///
    /// This function ensures that columns without build information are built first,
    /// and that columns with dependencies are built only after their dependencies have been satisfied.
    ///
    /// ## Arguments
    ///
    /// - `columns`: A vector of `BimmColumnSchema` representing the columns in the table.
    ///
    /// ## Returns
    ///
    /// A `Result<ColumnBuildOrder, String>` where:
    /// - `Ok(ColumnBuildOrder)` is the computed build order.
    /// - `Err(String)` is an error message if there are duplicate column names or circular dependencies.
    #[must_use]
    pub fn build_order(columns: &Vec<BimmColumnSchema>) -> Result<ColumnBuildOrder, String> {
        let mut col_names = std::collections::HashSet::new();
        for col in columns.clone() {
            if !col_names.insert(col.name.clone()) {
                return Err(format!("Duplicate column name: '{}'", col.name));
            }
        }

        let mut build_order = ColumnBuildOrder::default();

        for col in columns.clone() {
            match col.build_info.as_ref() {
                None => build_order.static_columns.push(col.name.clone()),
                Some(build_info) => {
                    for (_, cname) in &build_info.deps {
                        if !col_names.contains(cname) {
                            return Err(format!(
                                "Column '{}' depends on non-existent column '{}'",
                                col.name, cname
                            ));
                        }
                    }
                }
            }
        }

        while (build_order.static_columns.len() + build_order.topo_order.len()) < columns.len() {
            let mut progress = false;
            for col in columns.clone() {
                if let Some(build_info) = &col.build_info {
                    if build_order.topo_order.contains(&col.name) {
                        continue;
                    }

                    if build_info.deps.iter().all(|(_, cname)| {
                        build_order.static_columns.contains(cname)
                            || build_order.topo_order.contains(cname)
                    }) {
                        progress = true;
                        build_order.topo_order.push(col.name.clone());
                    }
                }
            }

            if !progress {
                return Err("Circular dependency detected in column build order".to_string());
            }
        }

        Ok(build_order)
    }
}

/// Bimm Table Schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BimmTableSchema {
    /// The columns in the table.
    pub columns: Vec<BimmColumnSchema>,
}

impl Index<usize> for BimmTableSchema {
    type Output = BimmColumnSchema;

    fn index(
        &self,
        index: usize,
    ) -> &Self::Output {
        &self.columns[index]
    }
}

impl IndexMut<usize> for BimmTableSchema {
    fn index_mut(
        &mut self,
        index: usize,
    ) -> &mut Self::Output {
        &mut self.columns[index]
    }
}

impl Index<&str> for BimmTableSchema {
    type Output = BimmColumnSchema;

    fn index(
        &self,
        index: &str,
    ) -> &Self::Output {
        let name = index.as_ref();
        self.column_index(name)
            .and_then(|idx| self.columns.get(idx))
            .expect("Column not found")
    }
}

impl IndexMut<&str> for BimmTableSchema {
    fn index_mut(
        &mut self,
        index: &str,
    ) -> &mut Self::Output {
        let name = index.as_ref();
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

impl BimmTableSchema {
    /// Creates a new `DataTableDescription` with the given columns.
    #[must_use]
    pub fn from_columns(columns: &[BimmColumnSchema]) -> Self {
        let columns = columns.to_vec();

        let _build_order = BimmColumnSchema::build_order(&columns).unwrap();

        Self { columns }
    }

    /// Returns the `ColumnBuildOrder` for the schema.
    pub fn build_order(&self) -> ColumnBuildOrder {
        BimmColumnSchema::build_order(&self.columns).unwrap()
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

        if let Some(build_info) = &column.build_info {
            for (_, cname) in &build_info.deps {
                if !self.column_index(cname).is_some() {
                    panic!(
                        "Column '{}' depends on non-existent column '{}'",
                        column.name, cname
                    );
                }
            }
        }
        self.columns.push(column);
    }

    /// Renames a column in the table description.
    pub fn rename_column(
        &mut self,
        old_name: &str,
        new_name: &str,
    ) -> Result<(), String> {
        if old_name == new_name {
            // No change needed
            return Ok(());
        }

        self.check_name(new_name)?;

        for col in &mut self.columns {
            if col.name == old_name {
                col.name = new_name.to_string();
            }
            if let Some(build_info) = &mut col.build_info {
                build_info.deps.values_mut().for_each(|cname| {
                    if *cname == old_name {
                        *cname = new_name.to_string();
                    }
                });
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
    /// use bimm::data::table::{BimmTableSchema, BimmColumnSchema};
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
    use super::*;
    use indoc::indoc;

    #[test]
    fn test_data_type_description() {
        let schema = BimmDataTypeDescription::new::<Option<i32>>();
        assert_eq!(schema.type_name, std::any::type_name::<Option<i32>>(),);
    }

    #[test]
    fn test_column_description() {
        let column = BimmColumnSchema::new::<Option<i32>>("abc");

        assert_eq!(column.name, "abc");
        assert_eq!(
            column.data_type,
            BimmDataTypeDescription::new::<Option<i32>>()
        );
    }

    #[test]
    fn test_schema() {
        let mut schema = BimmTableSchema::from_columns(&[BimmColumnSchema::new::<i32>("foo")]);

        // Index<usize>
        assert_eq!(schema[0].name, "foo");
        // IndexMut<usize>
        schema[0].description = Some("Overwritten".to_string());

        // Index<&str>
        assert_eq!(schema["foo"].name, "foo");
        // IndexMut<usize>
        schema["foo"].description = Some("An integer column".to_string());

        schema.add_column(BimmColumnSchema::new::<String>("bar").with_build_info(
            "build_bar",
            &[("source", "foo")],
            &[],
        ));

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
                      },
                      "build_info": {
                        "op": "build_bar",
                        "deps": {
                          "source": "foo"
                        }
                      }
                    }
                  ]
                }"#
            }
        );

        assert_eq!(
            schema.build_order(),
            ColumnBuildOrder {
                static_columns: vec!["foo".to_string()],
                topo_order: vec!["bar".to_string()],
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
                      },
                      "build_info": {
                        "op": "build_bar",
                        "deps": {
                          "source": "xxx"
                        }
                      }
                    }
                  ]
                }"#
            }
        );
    }

    #[test]
    #[should_panic(expected = "Duplicate column name: 'foo'")]
    fn conflicting_column_names_on_validate() {
        let _schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("foo"),
            BimmColumnSchema::new::<String>("foo"),
        ]);
    }

    #[test]
    #[should_panic(expected = "Column name 'foo' already exists")]
    fn conflicting_column_names_on_add() {
        let mut schema = BimmTableSchema::from_columns(&[BimmColumnSchema::new::<i32>("foo")]);
        schema.add_column(BimmColumnSchema::new::<String>("foo"));
    }
}
