use crate::core::identifiers;
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashSet};
use std::ops::{Index, IndexMut};

/// A serializable description of a data type.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct DataTypeDescription {
    /// The name of the data type.
    pub type_name: String,

    /// Type-specific extension data, serialized as JSON.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    #[serde(default)]
    pub extension: serde_json::Value,
}

impl DataTypeDescription {
    /// Creates a `DataTypeDescription` for a specific type `T`.
    ///
    /// This function uses `std::any::type_name` to get the type name of `T`.
    pub fn new<T>() -> Self {
        Self::from_type_name(std::any::type_name::<T>())
    }

    /// Creates a `DataTypeDescription` from a type name.
    pub fn from_type_name(type_name: &str) -> Self {
        DataTypeDescription {
            type_name: type_name.to_string(),
            extension: serde_json::Value::Null,
        }
    }

    /// Extends the data type description with a custom extension.
    ///
    /// ## Arguments
    ///
    /// - `extension`: The extension to attach to the data type description, serialized as JSON.
    ///
    /// ## Returns
    ///
    /// A new `DataTypeDescription` with the extension attached.
    pub fn with_extension<T>(
        self,
        extension: T,
    ) -> Self
    where
        T: Serialize,
    {
        DataTypeDescription {
            extension: serde_json::to_value(extension).expect("Failed to serialize extension"),
            ..self
        }
    }
}

/// A build plan for columns in a table schema.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct BuildPlan {
    /// The ID of the operator.
    pub operator_id: String,

    /// The description of the operator.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// Additional configuration for the operation, serialized as JSON.
    #[serde(skip_serializing_if = "serde_json::Value::is_null")]
    #[serde(default)]
    pub config: serde_json::Value,

    /// The input column bindings ``{parameter_name: column_name}``.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub inputs: BTreeMap<String, String>,

    /// The output column bindings ``{parameter_name: column_name}``.
    #[serde(skip_serializing_if = "BTreeMap::is_empty")]
    #[serde(default)]
    pub outputs: BTreeMap<String, String>,
}

impl BuildPlan {
    /// Creates a new `ColumnBuildPlan` with the given operator spec.
    pub fn for_operator<S>(id: S) -> Self
    where
        S: AsRef<str>,
    {
        BuildPlan {
            operator_id: id.as_ref().to_string(),
            description: None,
            config: serde_json::Value::Null,
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
        }
    }

    /// Extends the build plan with a description.
    ///
    /// ## Arguments
    ///
    /// - `description`: The description to attach to the build plan.
    ///
    /// ## Returns
    ///
    /// A new `ColumnBuildPlan` with the description attached.
    pub fn with_description(
        self,
        description: &str,
    ) -> Self {
        BuildPlan {
            description: Some(description.to_string()),
            ..self
        }
    }

    /// Extends the build plan with a configuration.
    ///
    /// ## Arguments
    ///
    /// - `config`: The configuration to attach to the build plan, serialized as JSON.
    ///
    /// ## Returns
    ///
    /// A new `ColumnBuildPlan` with the configuration attached.
    pub fn with_config<T>(
        self,
        config: T,
    ) -> Self
    where
        T: Serialize,
    {
        BuildPlan {
            config: serde_json::to_value(config).expect("Failed to serialize config"),
            ..self
        }
    }

    fn build_name_map(assoc: &[(&str, &str)]) -> BTreeMap<String, String> {
        let mut btree = BTreeMap::new();
        for (param, column) in assoc {
            identifiers::check_ident(param).expect("Invalid parameter name");
            identifiers::check_ident(column).expect("Invalid column name");
            btree.insert(param.to_string(), column.to_string());
        }
        btree
    }

    /// Extends the build plan with input columns.
    ///
    /// ## Arguments
    ///
    /// - `inputs`: A slice of tuples where each tuple contains a parameter name and a column name.
    ///
    /// ## Returns
    ///
    /// A new `ColumnBuildPlan` with the input columns attached.
    pub fn with_inputs(
        self,
        inputs: &[(&str, &str)],
    ) -> Self {
        BuildPlan {
            inputs: Self::build_name_map(inputs),
            ..self
        }
    }

    /// Extends the build plan with output columns.
    ///
    /// ## Arguments
    ///
    /// - `outputs`: A slice of tuples where each tuple contains a parameter name and a column name.
    ///
    /// ## Returns
    ///
    /// A new `ColumnBuildPlan` with the output columns attached.
    pub fn with_outputs(
        self,
        outputs: &[(&str, &str)],
    ) -> Self {
        BuildPlan {
            outputs: Self::build_name_map(outputs),
            ..self
        }
    }
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
    pub build_plans: Vec<BuildPlan>,
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

impl TableSchema {
    fn check_graph(
        columns: &[ColumnSchema],
        plans: &[BuildPlan],
    ) -> Result<(Vec<String>, Vec<BuildPlan>), String> {
        let column_names: Vec<String> = columns.iter().map(|c| c.name.clone()).collect();

        let mut base_columns: Vec<String> = column_names.clone();
        for plan in plans {
            for (_, cname) in plan.outputs.iter() {
                base_columns.retain(|c| c != cname);
            }
        }

        let base_columns = base_columns;

        let mut scheduled_columns = base_columns.clone();
        let mut plan_order = Vec::new();

        // TODO: build reified topo dep-graph;
        // support printing, querying, cycle detection, etc.

        loop {
            let mut progress = false;

            for (idx, plan) in plans.iter().enumerate() {
                if plan_order.contains(&idx) {
                    // This plan is already scheduled.
                    continue;
                }
                if plan
                    .inputs
                    .values()
                    .all(|cname| scheduled_columns.contains(cname))
                {
                    // All inputs are scheduled, we can schedule this plan.
                    for (_, cname) in plan.outputs.iter() {
                        if !scheduled_columns.contains(cname) {
                            scheduled_columns.push(cname.clone());
                        }
                    }
                    progress = true;
                    plan_order.push(idx);
                } else {
                    // Not all inputs are scheduled, skip this plan.
                    continue;
                }
            }

            if !progress {
                // No progress was made, we are done.
                break;
            }
        }

        if scheduled_columns.len() != column_names.len() {
            return Err(format!(
                "Not all columns are scheduled: expected {}, got {}",
                column_names.len(),
                scheduled_columns.len()
            ));
        }

        let order: Vec<BuildPlan> = plan_order.iter().map(|&idx| plans[idx].clone()).collect();

        Ok((base_columns, order))
    }

    /// Compute the build order for the table schema.
    ///
    /// This function checks the build plans and their dependencies to determine the order in which they should be executed.
    ///
    /// ## Returns
    ///
    /// A `Result<(Vec<String>, Vec<BuildPlan>), String>` where:
    /// - `Ok((Vec<String>, Vec<BuildPlan>))` contains the base columns and the ordered build plans.
    /// - `Err(String)` contains an error message if the build order cannot be determined.
    pub fn build_order(&self) -> Result<(Vec<String>, Vec<BuildPlan>), String> {
        Self::check_graph(&self.columns, &self.build_plans)
    }

    /// Computes the build order for a target set of columns, given the extant columns.
    ///
    /// This function determines which build plans are needed to produce the target columns, based on the extant columns.
    ///
    /// ## Arguments
    ///
    /// - `extant_columns`: A vector of column names that already exist in the table.
    /// - `target_columns`: A vector of column names that we want to build.
    ///
    /// ## Returns
    ///
    /// A `Result<Vec<BuildPlan>, String>` where:
    /// - `Ok(Vec<BuildPlan>)` contains the ordered build plans needed to produce the target columns.
    /// - `Err(String)` contains an error message if the build order cannot be determined.
    pub fn target_build_order(
        &self,
        extant_columns: &[&str],
        target_columns: &[&str],
    ) -> Result<Vec<BuildPlan>, String> {
        let (_, plans) = self.build_order()?;

        let mut needed: HashSet<String> = HashSet::new();
        for cname in target_columns {
            if !extant_columns.contains(cname) {
                needed.insert(cname.to_string());
            }
        }

        let mut order: Vec<BuildPlan> = Vec::new();

        for plan in plans.iter().rev() {
            if plan
                .outputs
                .iter()
                .any(|(_, cname)| needed.contains(cname.as_str()))
            {
                order.push(plan.clone());

                for (_, cname) in plan.inputs.iter() {
                    if !extant_columns.contains(&cname.as_str()) {
                        needed.insert(cname.clone());
                    }
                }
            }
        }

        order.reverse();

        Ok(order)
    }

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
        identifiers::check_ident(name)?;

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
        plan: BuildPlan,
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

        {
            let mut plans = self.build_plans.clone();
            plans.push(plan.clone());
            Self::check_graph(&self.columns, plans.as_slice())?;
        }

        self.build_plans.push(plan);
        Ok(())
    }

    /// Adds a build plan and its output columns to the table description.
    ///
    /// ## Arguments
    ///
    /// - `plan`: The build plan to add.
    /// - `output_info`: A slice of tuples where each tuple contains the output column name, its data type, and a description.
    ///
    /// ## Returns
    ///
    /// A `Result<(), String>` where:
    /// - `Ok(())` indicates success.
    /// - `Err(String)` contains an error message if the operation fails.
    pub fn add_build_plan_and_outputs(
        &mut self,
        plan: BuildPlan,
        output_info: &[(String, DataTypeDescription, Option<String>)],
    ) -> Result<(), String> {
        // Now add the output columns.
        for (pname, data_type, description) in output_info {
            let cname = plan.outputs.get(pname).expect("Output column not found");
            let column = ColumnSchema {
                name: cname.to_string(),
                description: description.clone(),
                data_type: data_type.clone(),
            };
            self.add_column(column);
        }

        self.add_build_plan(plan)
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
    /// use bimm_firehose::core::{TableSchema, ColumnSchema};
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
