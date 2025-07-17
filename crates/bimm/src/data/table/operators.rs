use crate::data::table::{AnyArc, BimmRow, BimmTableSchema, BuildPlan};
use crate::data::table::{BimmDataTypeDescription, OperatorSpec};
use std::any::Any;
use std::collections::BTreeMap;
use std::fmt::Debug;
use std::sync::Arc;

/// Factory trait for building operators in a `BuildPlan`.
pub trait BuildOperatorFactory: Debug {
    /// Initialize an operator based on the provided specification and input/output types.
    ///
    /// # Arguments
    ///
    /// * `spec` - The specification of the operator to be initialized.
    /// * `input_types` - A map of input names to their data type descriptions.
    /// * `output_types` - A map of output names to their data type descriptions.
    ///
    /// # Returns
    ///
    /// A result containing a boxed `BuildOperator` if initialization is successful, or an error message if it fails.
    fn init_operator(
        &self,
        spec: &OperatorSpec,
        input_types: &BTreeMap<String, BimmDataTypeDescription>,
        output_types: &BTreeMap<String, BimmDataTypeDescription>,
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
    pub fn new(namespace: String) -> Self {
        NamespaceOperatorFactory {
            namespace,
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

    /// Register multiple operator factories for operations within the namespace.
    ///
    /// # Arguments
    ///
    /// * `operations` - A slice of tuples, where each tuple contains the operation name and the corresponding operator factory.
    pub fn add_operations(
        &mut self,
        operations: &[(&str, Arc<dyn BuildOperatorFactory>)],
    ) {
        for (operation, factory) in operations {
            self.add_operation(operation, factory.clone());
        }
    }
}

impl BuildOperatorFactory for NamespaceOperatorFactory {
    fn init_operator(
        &self,
        spec: &OperatorSpec,
        input_types: &BTreeMap<String, BimmDataTypeDescription>,
        output_types: &BTreeMap<String, BimmDataTypeDescription>,
    ) -> Result<Box<dyn BuildOperator>, String> {
        if let Some(factory) = self.operations.get(&spec.id.name) {
            factory.init_operator(spec, input_types, output_types)
        } else {
            Err(format!(
                "No operator factory registered for operation '{}' in namespace '{}'",
                spec.id.name, self.namespace
            ))
        }
    }
}

/// Factory for dispatching operator factories based on namespaces.
#[derive(Debug)]
pub struct NamespaceMapOperatorFactory {
    /// A map of namespaces to their corresponding operator factories.
    pub factories: BTreeMap<String, Arc<dyn BuildOperatorFactory>>,
}

impl Default for NamespaceMapOperatorFactory {
    fn default() -> Self {
        NamespaceMapOperatorFactory::new()
    }
}

impl NamespaceMapOperatorFactory {
    /// Create a new `NamespaceMapOperatorFactory`.
    pub fn new() -> Self {
        NamespaceMapOperatorFactory {
            factories: BTreeMap::new(),
        }
    }

    /// Register an operator factory for a specific namespace.
    pub fn add_namespace(
        &mut self,
        namespace: &str,
        factory: Arc<dyn BuildOperatorFactory>,
    ) {
        if self.factories.contains_key(namespace) {
            panic!("Operator factory for namespace '{namespace}' already registered");
        }
        self.factories.insert(namespace.to_string(), factory);
    }
}

impl BuildOperatorFactory for NamespaceMapOperatorFactory {
    fn init_operator(
        &self,
        spec: &OperatorSpec,
        input_types: &BTreeMap<String, BimmDataTypeDescription>,
        output_types: &BTreeMap<String, BimmDataTypeDescription>,
    ) -> Result<Box<dyn BuildOperator>, String> {
        if let Some(factory) = self.factories.get(&spec.id.namespace) {
            factory.init_operator(spec, input_types, output_types)
        } else {
            Err(format!(
                "No operator factory registered for namespace '{}'",
                spec.id.namespace
            ))
        }
    }
}

type BuildInputRefMap<'a> = BTreeMap<&'a str, Option<&'a dyn Any>>;
type BuildOutputArcMap<'a> = BTreeMap<&'a str, Option<AnyArc>>;

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

pub struct BoundPlanBuilder {
    /// The table schema that this operator is bound to.
    pub table_schema: BimmTableSchema,

    /// A reference to the build plan that this operator is part of.
    pub build_plan: BuildPlan,

    /// Maps from input parameter names to their slot indices in the input row.
    input_slot_map: BTreeMap<String, usize>,

    /// Maps from output parameter names to their slot indices in the output row.
    output_slot_map: BTreeMap<String, usize>,

    /// The operator that this builder wraps.
    operator: Box<dyn BuildOperator>,
}

impl BoundPlanBuilder {
    pub fn bind_plan<F>(
        table_schema: &BimmTableSchema,
        build_plan: &BuildPlan,
        factory: &F,
    ) -> Result<BoundPlanBuilder, String>
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

        let operator = factory.init_operator(&build_plan.operator, &input_types, &output_types)?;

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

        Ok(BoundPlanBuilder {
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
        rows: &mut [BimmRow],
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
            for (&pname, value) in outputs.iter() {
                if let Some(&index) = self.output_slot_map.get(pname) {
                    row.set_slot(index, value.clone());
                } else {
                    return Err(format!(
                        "Output parameter '{pname}' not found in output slot map"
                    ));
                }
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {

}
