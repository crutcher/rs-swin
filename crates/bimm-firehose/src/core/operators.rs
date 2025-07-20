use crate::core::{AnyArc, BuildPlan, DataTypeDescription, Row, TableSchema};
use std::any::Any;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

type BuildInputRefMap<'a> = BTreeMap<&'a str, Option<&'a dyn Any>>;
type BuildOutputArcMap<'a> = BTreeMap<String, Option<AnyArc>>;

/// Implementation of a `BuildPlan` operator.
pub trait BuildOperator {
    /// Get the effective batch size for the operator.
    ///
    /// The default implementation returns 1, indicating that the operator processes one row at a time.
    fn effective_batch_size(&self) -> usize {
        1
    }

    /// Apply the operator to a batch of inputs.
    ///
    /// The default implementation iterates over each input row and applies the operator individually;
    /// with no batch acceleration.
    ///
    /// Implementations can override this method to provide batch processing capabilities,
    /// and should update the `effective_batch_size` method accordingly to reflect the batch size used.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A slice of maps, where each map contains input names and their corresponding values.
    ///
    /// # Returns
    ///
    /// A result containing a vector of maps, where each map contains output names and their corresponding values.
    fn apply_batch(
        &self,
        inputs: &[BuildInputRefMap],
    ) -> Result<Vec<BuildOutputArcMap>, String> {
        let mut results = Vec::new();
        for row in inputs {
            let result = self.apply(row)?;
            results.push(result);
        }
        Ok(results)
    }

    /// Apply the operator to the provided inputs.
    ///
    /// # Arguments
    ///
    /// * `inputs` - A map of input names to their values, where the values are wrapped in `Option<&dyn Any>`.
    ///
    /// # Returns
    ///
    /// A result containing a map of output names to their values, where the values are also wrapped in `Option<&dyn Any>`.
    fn apply(
        &self,
        inputs: &BuildInputRefMap,
    ) -> Result<BuildOutputArcMap, String>;
}

/// A runner for a column operator that applies a `BuildOperator` to rows in a table schema.
pub struct ColumnBuilder {
    /// The table schema that this operator is bound to.
    pub table_schema: TableSchema,

    /// A reference to the build plan that this operator is part of.
    pub build_plan: BuildPlan,

    /// Maps from input parameter names to their slot indices in the input row.
    input_slot_map: BTreeMap<String, usize>,

    /// Maps from output parameter names to their slot indices in the output row.
    output_slot_map: BTreeMap<String, usize>,

    /// The operator that this builder wraps.
    operator: Box<dyn BuildOperator>,
}

impl Debug for ColumnBuilder {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if !f.alternate() {
            f.debug_struct("ColumnBuilder")
                .field("id", &self.build_plan.operator)
                .field("inputs", &self.build_plan.inputs)
                .field("outputs", &self.build_plan.outputs)
                .finish()
        } else {
            f.debug_struct("ColumnBuilder")
                .field("build_plan", &self.build_plan)
                .finish()
        }
    }
}

impl ColumnBuilder {
    /// Create a new `BoundPlanBuilder` by binding a `BuildPlan` to a `BimmTableSchema`.
    ///
    /// # Arguments
    ///
    /// * `table_schema` - The schema of the table to which this plan is bound.
    /// * `build_plan` - The build plan that describes the operator and its inputs/outputs.
    /// * `factory` - A factory that can create the operator based on the build plan.
    ///
    /// # Returns
    ///
    /// A result containing a `BoundPlanBuilder` if successful, or an error message if the binding fails.
    #[must_use]
    pub fn bind_plan<F>(
        table_schema: &TableSchema,
        build_plan: &BuildPlan,
        factory: &F,
    ) -> Result<ColumnBuilder, String>
    where
        F: BuildOperatorFactory,
    {
        let table_schema = table_schema.clone();
        let build_plan = build_plan.clone();

        let input_types = build_plan
            .inputs
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.clone(),
                    table_schema[cname.as_ref()].data_type.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let output_types = build_plan
            .outputs
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.clone(),
                    table_schema[cname.as_ref()].data_type.clone(),
                )
            })
            .collect::<BTreeMap<_, _>>();

        let operator = factory.init_operator(&build_plan, &input_types, &output_types)?;

        let input_slot_map = build_plan
            .inputs
            .iter()
            .map(|(pname, cname)| (pname.clone(), table_schema.column_index(cname).unwrap()))
            .collect::<BTreeMap<_, _>>();

        let output_slot_map = build_plan
            .outputs
            .iter()
            .map(|(pname, cname)| (pname.clone(), table_schema.column_index(cname).unwrap()))
            .collect::<BTreeMap<_, _>>();

        Ok(ColumnBuilder {
            table_schema,
            build_plan,
            input_slot_map,
            output_slot_map,
            operator,
        })
    }

    /// Get the effective batch size for the operator.
    pub fn effective_batch_size(&self) -> usize {
        self.operator.effective_batch_size()
    }

    /// Apply the operator to a batch of rows.
    ///
    /// # Arguments
    ///
    /// * `rows` - A mutable slice of `BimmRow` instances that will be processed by the operator.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error message if the operation fails.
    pub fn apply_batch(
        &self,
        rows: &mut [Row],
    ) -> Result<(), String> {
        let batch_inputs: Vec<BuildInputRefMap> = rows
            .iter()
            .map(|row| {
                self.input_slot_map
                    .iter()
                    .map(|(pname, &index)| (pname.as_str(), row.get_untyped_slot(index)))
                    .collect::<BuildInputRefMap>()
            })
            .collect::<Vec<_>>();

        let batch_outputs = self.operator.apply_batch(&batch_inputs)?;

        for (idx, outputs) in batch_outputs.iter().enumerate() {
            let row = &mut rows[idx];
            for (pname, value) in outputs.iter() {
                row.set_slot(self.output_slot_map[pname], value.clone());
            }
        }

        Ok(())
    }
}

/// Factory trait for building operators in a `BuildPlan`.
pub trait BuildOperatorFactory: Debug {
    /// Initialize an operator based on the provided specification and input/output types.
    ///
    /// # Arguments
    ///
    /// * `build_plan` - The build plan of the operator to be initialized.
    /// * `input_types` - A map of input names to their data type descriptions.
    /// * `output_types` - A map of output names to their data type descriptions.
    ///
    /// # Returns
    ///
    /// A result containing a boxed `BuildOperator` if initialization is successful, or an error message if it fails.
    fn init_operator(
        &self,
        build_plan: &BuildPlan,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<Box<dyn BuildOperator>, String>;
}

/// A factory for building operators within a specific namespace.
#[derive(Debug)]
pub struct NamespaceOperatorFactory {
    /// The namespace of the operator factory.
    pub namespace: String,

    /// The operator factory to be used for building operators.
    pub operations: BTreeMap<String, Arc<dyn BuildOperatorFactory>>,
}

impl NamespaceOperatorFactory {
    /// Create a new `NamespaceOperatorFactory`.
    ///
    /// # Arguments
    ///
    /// * `namespace` - The namespace for the operator factory.
    /// * `factory` - The operator factory to be used for building operators.
    pub fn new(namespace: &str) -> Self {
        NamespaceOperatorFactory {
            namespace: namespace.to_string(),
            operations: Default::default(),
        }
    }

    /// Register an operator factory for a specific operation within the namespace.
    ///
    /// # Arguments
    ///
    /// * `operation` - The name of the operation to register.
    ///
    /// * `factory` - The operator factory to be registered for the operation.
    pub fn add_operation(
        &mut self,
        operation: &str,
        factory: Arc<dyn BuildOperatorFactory>,
    ) {
        if self.operations.contains_key(operation) {
            panic!(
                "Operator factory for operation '{}' already registered in namespace '{}'",
                operation, self.namespace
            );
        }
        self.operations.insert(operation.to_string(), factory);
    }
}

impl BuildOperatorFactory for NamespaceOperatorFactory {
    fn init_operator(
        &self,
        build_plan: &BuildPlan,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<Box<dyn BuildOperator>, String> {
        if let Some(factory) = self.operations.get(&build_plan.operator.name) {
            factory.init_operator(build_plan, input_types, output_types)
        } else {
            Err(format!(
                "No operator factory registered for operation '{}' in namespace '{}'",
                build_plan.operator.name, self.namespace
            ))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::{
        AnyArc, BuildPlan, ColumnSchema, DataTypeDescription, OperatorId, RowBatch, TableSchema,
    };
    use indoc::formatdoc;
    use serde::{Deserialize, Serialize};
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::fmt::Debug;
    use std::sync::Arc;

    const EXAMPLE_NAMESPACE: &str = "example";

    // A simple add operator that adds inputs and applies a bias
    #[derive(Debug)]
    struct AddOperatorFactory {}

    impl BuildOperatorFactory for AddOperatorFactory {
        fn init_operator(
            &self,
            build_plan: &BuildPlan,
            input_types: &BTreeMap<String, DataTypeDescription>,
            output_types: &BTreeMap<String, DataTypeDescription>,
        ) -> Result<Box<dyn BuildOperator>, String> {
            // Verify that all inputs are i32
            let expected_type = DataTypeDescription::new::<i32>();
            for (name, data_type) in input_types {
                if data_type.type_name != expected_type.type_name {
                    return Err(format!(
                        "Input '{}' must be of type i32, found {:?}",
                        name, data_type.type_name
                    ));
                }
            }

            if output_types.len() != 1 || output_types.get("result").is_none() {
                return Err("AddOperator must have a single output named 'result'".to_string());
            }
            if output_types["result"].type_name != expected_type.type_name {
                return Err("Output 'result' must be of type i32".to_string());
            }

            let op: AddOperator = serde_json::from_value(build_plan.config.clone())
                .map_err(|e| format!("Failed to parse operator config: {e}"))?;

            Ok(Box::new(op))
        }
    }

    #[derive(Debug, Serialize, Deserialize)]
    struct AddOperator {
        bias: i32,
    }

    impl BuildOperator for AddOperator {
        fn apply(
            &self,
            inputs: &BTreeMap<&str, Option<&dyn Any>>,
        ) -> Result<BTreeMap<String, Option<AnyArc>>, String> {
            let sum: i32 = inputs
                .values()
                .map(|v| v.unwrap().downcast_ref::<i32>().unwrap())
                .sum();

            // Add the bias
            let result: i32 = sum + self.bias;

            // Return the result as a single output
            let mut outputs = BTreeMap::new();
            outputs.insert("result".to_string(), Some(Arc::new(result) as AnyArc));

            Ok(outputs)
        }
    }

    #[test]
    #[should_panic(expected = "Input 'a' must be of type i32")]
    fn test_bad_input_data_type() {
        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<String>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(OperatorId::new(EXAMPLE_NAMESPACE, "add"))
                    .with_description("Adds inputs with a bias")
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("a", "a"), ("b", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let factory = AddOperatorFactory {};

        let _builder = ColumnBuilder::bind_plan(&schema, &schema.build_plans[0], &factory).unwrap();
    }

    #[test]
    #[should_panic(expected = "Output 'result' must be of type i32")]
    fn test_bad_output_data_type() {
        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<String>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(OperatorId::new(EXAMPLE_NAMESPACE, "add"))
                    .with_description("Adds inputs with a bias")
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("x", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let factory = AddOperatorFactory {};

        let _builder = ColumnBuilder::bind_plan(&schema, &schema.build_plans[0], &factory).unwrap();
    }

    #[test]
    fn test_simple_op() {
        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(OperatorId::new(EXAMPLE_NAMESPACE, "add"))
                    .with_description("Adds inputs with a bias")
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let factory = AddOperatorFactory {};

        let builder = ColumnBuilder::bind_plan(&schema, &schema.build_plans[0], &factory).unwrap();

        assert_eq!(
            format!("{builder:?}"),
            "ColumnBuilder { id: OperatorId { namespace: \"example\", name: \"add\" }, inputs: {\"x\": \"a\", \"y\": \"b\"}, outputs: {\"result\": \"c\"} }"
        );

        assert_eq!(
            format!("{builder:#?}"),
            formatdoc! {r#"
               ColumnBuilder {{
                   build_plan: BuildPlan {{
                       operator: OperatorId {{
                           namespace: "example",
                           name: "add",
                       }},
                       description: Some(
                           "Adds inputs with a bias",
                       ),
                       config: Object {{
                           "bias": Number(10),
                       }},
                       inputs: {{
                           "x": "a",
                           "y": "b",
                       }},
                       outputs: {{
                           "result": "c",
                       }},
                   }},
               }}"#,
            }
        );

        assert_eq!(builder.effective_batch_size(), 1);

        assert_eq!(
            builder.build_plan.operator,
            OperatorId {
                namespace: EXAMPLE_NAMESPACE.to_string(),
                name: "add".to_string()
            }
        );

        let mut batch = RowBatch::with_size(Arc::new(schema.clone()), 2);
        batch[0].set_columns(&schema, &["a", "b"], [Arc::new(10), Arc::new(20)]);
        batch[1].set_columns(&schema, &["a", "b"], [Arc::new(-5), Arc::new(2)]);

        builder.apply_batch(batch.rows.as_mut_slice()).unwrap();

        assert_eq!(batch[0].get_column::<i32>(&schema, "c").unwrap(), &40);
        assert_eq!(batch[1].get_column::<i32>(&schema, "c").unwrap(), &7);
    }

    #[test]
    fn test_namespace_factory() {
        let mut factory = NamespaceOperatorFactory::new(EXAMPLE_NAMESPACE);

        // Create a table schema with a build plan for the add operator
        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(OperatorId::new(EXAMPLE_NAMESPACE, "add"))
                    .with_description("Adds inputs with a bias")
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("x", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        // The operation isn't registered in the factory, so this should fail.
        assert_eq!(
            ColumnBuilder::bind_plan(&schema, &schema.build_plans[0], &factory).unwrap_err(),
            "No operator factory registered for operation 'add' in namespace 'example'".to_string()
        );

        factory.add_operation("add", Arc::new(AddOperatorFactory {}));

        // Bind the plan using the namespace map factory
        let builder = ColumnBuilder::bind_plan(&schema, &schema.build_plans[0], &factory).unwrap();

        assert_eq!(builder.effective_batch_size(), 1);
    }
}
