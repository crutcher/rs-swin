use crate::core::operations::environment::OpEnvironment;
use crate::core::operations::signature::FirehoseOperatorSignature;
use crate::core::schema::{BuildPlan, ColumnSchema, FirehoseTableSchema};
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A builder for constructing a call to an operator in a `BuildPlan`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct OperationPlan {
    /// The ID of the operator to be called.
    pub operator_id: String,

    /// A map from formal parameter names to column names for inputs.
    pub inputs: BTreeMap<String, String>,

    /// A map from formal parameter names to column names for outputs.
    pub outputs: BTreeMap<String, String>,

    /// An optional configuration for the call, serialized as JSON.
    pub config: Option<serde_json::Value>,
}

impl OperationPlan {
    /// Creates a new `CallBuilder` for the specified operator ID.
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The ID of the operator to be called.
    pub fn for_operation_id(operator_id: &str) -> Self {
        Self {
            operator_id: operator_id.to_string(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            config: None,
        }
    }

    /// Adds an input parameter to the call builder.
    ///
    /// # Arguments
    ///
    /// * `pname` - The name of the input parameter.
    /// * `cname` - The column name in the input table.
    ///
    /// # Returns
    ///
    /// The same `CallBuilder` instance with the input parameter added.
    pub fn with_input(
        mut self,
        pname: &str,
        cname: &str,
    ) -> Self {
        if self.inputs.contains_key(pname) {
            panic!("Input parameter '{pname}' already exists.");
        }
        self.inputs.insert(pname.to_string(), cname.to_string());
        self
    }

    /// Adds an output parameter to the call builder.
    ///
    /// # Arguments
    ///
    /// * `pname` - The name of the output parameter.
    /// * `cname` - The column name in the output table.
    ///
    /// # Returns
    ///
    /// The same `CallBuilder` instance with the output parameter added.
    pub fn with_output(
        mut self,
        pname: &str,
        cname: &str,
    ) -> Self {
        if self.outputs.contains_key(pname) {
            panic!("Output parameter '{pname}' already exists.");
        }
        self.outputs.insert(pname.to_string(), cname.to_string());
        self
    }

    /// Adds a configuration to the call builder.
    ///
    /// The configuration is serialized to JSON and stored in the call.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to be added, which must implement `Serialize`.
    ///
    /// # Returns
    ///
    /// The same `CallBuilder` instance with the configuration added.
    pub fn with_config<T>(
        mut self,
        config: T,
    ) -> Self
    where
        T: Serialize,
    {
        self.config = Some(serde_json::to_value(config).expect("Failed to serialize config"));
        self
    }

    /// Binds the context to a specific operator signature.
    ///
    /// This does not validate the plan and signature against any schema.
    ///
    /// # Arguments
    ///
    /// * `signature` - The operator signature to bind to the context.
    ///
    /// # Returns
    ///
    /// A result containing an `OperationInitSignatureContext` if successful, or an error message if the operation fails.
    pub fn plan_for_signature(
        self,
        signature: &FirehoseOperatorSignature,
    ) -> Result<(BuildPlan, BTreeMap<String, ColumnSchema>), String> {
        let mut plan = BuildPlan::for_operator(self.operator_id);
        plan.inputs = self.inputs.clone();
        plan.outputs = self.outputs.clone();

        if let Some(description) = &signature.description {
            plan = plan.with_description(description);
        }
        if let Some(config) = &self.config {
            plan = plan.with_config(config);
        }

        let output_cols = signature.output_column_schemas_for_plan(&plan)?;

        Ok((plan, output_cols))
    }

    /// Applies the operation planner to a table schema and environment.
    ///
    /// # Arguments
    ///
    /// * `schema` - The mutable reference to the table schema to which the operation will be applied.
    /// * `env` - The environment that can create the operator based on the build plan.
    ///
    /// # Returns
    ///
    /// A result containing a `BuildPlan` if successful, or an error message if the operation fails.
    pub fn apply_to_schema(
        self,
        schema: &mut FirehoseTableSchema,
        env: &dyn OpEnvironment,
    ) -> Result<BuildPlan, String> {
        env.apply_plan_to_schema(schema, self)
    }
}
