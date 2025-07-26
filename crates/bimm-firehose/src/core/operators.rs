use crate::core::{AnyArc, BuildPlan, DataTypeDescription, Row, TableSchema};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::{BTreeMap, HashSet};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

/// Define a self-referential operator ID.
///
/// The id will be defined as a static string constant that refers to its own namespace path.
///
/// ## Arguments
///
/// * `$name`: The name of the operator ID to define.
///
#[macro_export]
macro_rules! define_operator_id {
    ($name:ident) => {
        /// Self-referential operator ID.
        pub static $name: &str = concat!(module_path!(), "::", stringify!($name),);
    };
}

/// Struct describing a name to constructor for an operator builder.
///
/// Used by `register_default_operator_builder!` to register operator builders
/// which do not require any additional configuration or parameters.
#[derive(Clone)]
pub struct ColumnBuildOperationBuilderRegistration {
    /// The operator ID.
    pub operator_id: &'static str,

    /// The builder.
    pub supplier: fn() -> Arc<dyn ColumnBuildOperatorBuilder>,
}

impl Debug for ColumnBuildOperationBuilderRegistration {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("ColumnBuildOperationBuilderRegistration")
            .field("operator_id", &self.operator_id)
            .finish()
    }
}

inventory::collect! {
    ColumnBuildOperationBuilderRegistration
}

impl ColumnBuildOperationBuilderRegistration {
    /// Creates a new registration.
    pub const fn new(
        operator_id: &'static str,
        supplier: fn() -> Arc<dyn ColumnBuildOperatorBuilder>,
    ) -> Self {
        Self {
            operator_id,
            supplier,
        }
    }

    /// Returns the builder.
    pub fn get_builder(&self) -> Arc<dyn ColumnBuildOperatorBuilder> {
        (self.supplier)()
    }
}

/// Macro to register a default operator builder.
///
/// Builders which do not require runtime configuration can be registered
/// using this macro; and collected globally using `list_default_operator_builders`.
///
/// You can also collect a default environment with all registered builders
/// using `new_default_operator_environment`.
#[macro_export]
macro_rules! register_default_operator_builder {
    ($name:ident, $constructor:expr) => {
        inventory::submit! {
            $crate::core::ColumnBuildOperationBuilderRegistration::new(
                $name,
                || ($constructor)(),
            )
        }
    };
}

/// Lists all default operator builders constructors registered in the inventory.
pub fn list_default_operator_builders() -> Vec<&'static ColumnBuildOperationBuilderRegistration> {
    inventory::iter::<ColumnBuildOperationBuilderRegistration>
        .into_iter()
        .collect()
}

/// Build the default environment.
///
/// This constructs a `MapOpEnvironment` and adds all operator builders
/// registered with `bimm_firehose::register_default_operator_builder!`.
///
/// Each call `build_default_environment` will create a new mutable environment.
pub fn new_default_operator_environment() -> MapOpEnvironment {
    let mut env = MapOpEnvironment::new();

    for reg in list_default_operator_builders() {
        env.add_binding(reg.get_builder()).unwrap();
    }

    env
}

/// Defines the arity (requirement level) of a parameter
#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ParameterArity {
    /// The parameter is required and must be provided.
    #[default]
    Required,

    /// The parameter is optional and may be omitted.
    Optional,
}

/// Defines a single parameter specification
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct ParameterSpec {
    /// The name of the parameter.
    pub name: String,

    /// An optional description of the parameter.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub description: Option<String>,

    /// The data type of the parameter.
    pub data_type: DataTypeDescription,

    /// The arity of the parameter, indicating whether it is required or optional.
    pub arity: ParameterArity,
}

impl ParameterSpec {
    /// Creates a new `ParameterSpec` with the given name and data type.
    ///
    /// The type `T` is used to infer the data type of the parameter.
    ///
    /// ## Parameters
    ///
    /// - `name`: The name of the parameter.
    /// - `T`: The type of the parameter, which is used to determine the data type description.
    ///
    /// ## Returns
    ///
    /// A new `ParameterSpec` instance with the specified name, data type, and required arity.
    pub fn new<T>(name: &str) -> Self {
        Self {
            name: name.to_string(),
            description: None,
            data_type: DataTypeDescription::new::<T>(),
            arity: ParameterArity::Required,
        }
    }

    /// Extends the parameter specification with a description.
    pub fn with_description(
        self,
        description: impl Into<String>,
    ) -> Self {
        Self {
            name: self.name,
            description: Some(description.into()),
            data_type: self.data_type,
            arity: self.arity,
        }
    }

    /// Extends the parameter specification with a data type.
    pub fn with_arity(
        self,
        arity: ParameterArity,
    ) -> Self {
        Self {
            name: self.name,
            description: self.description,
            data_type: self.data_type,
            arity,
        }
    }
}

/// Defines the complete input/output specification for an operator
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorSpec {
    /// The identifier for the operator.
    pub operator_id: Option<String>,

    /// Optional description.
    pub description: Option<String>,

    /// A list of input parameters for the operator.
    pub inputs: Vec<ParameterSpec>,

    /// A list of output parameters for the operator.
    pub outputs: Vec<ParameterSpec>,
}

impl Default for OperatorSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl OperatorSpec {
    /// Creates a new `OperatorSpec` with no inputs or outputs.
    pub const fn new() -> Self {
        Self {
            operator_id: None,
            description: None,
            inputs: Vec::new(),
            outputs: Vec::new(),
        }
    }

    /// Extends the operator specification with an operator ID.
    pub fn with_operator_id(
        self,
        operator_id: &str,
    ) -> Self {
        Self {
            operator_id: Some(operator_id.into()),
            description: self.description,
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    /// Extends the operator specification with a description.
    pub fn with_description(
        self,
        description: &str,
    ) -> Self {
        Self {
            operator_id: self.operator_id,
            description: Some(description.to_string()),
            inputs: self.inputs,
            outputs: self.outputs,
        }
    }

    /// Extends the operator specification with an input parameter.
    pub fn with_input(
        self,
        spec: ParameterSpec,
    ) -> Self {
        if self.inputs.iter().any(|prev| prev.name == spec.name) {
            panic!("Duplicate parameter '{}'.", spec.name);
        }

        let mut inputs = self.inputs;
        inputs.push(spec);
        Self {
            operator_id: self.operator_id,
            description: self.description,
            inputs,
            outputs: self.outputs,
        }
    }

    /// Generate an output plan suitable for `.add_build_plan_and_outputs()`.
    pub fn output_plan(&self) -> Vec<(String, DataTypeDescription, Option<String>)> {
        self.outputs
            .iter()
            .map(|spec| {
                (
                    spec.name.clone(),
                    spec.data_type.clone(),
                    spec.description.clone(),
                )
            })
            .collect()
    }

    /// Extends the operator specification with an output parameter.
    pub fn with_output(
        self,
        spec: ParameterSpec,
    ) -> Self {
        if self.outputs.iter().any(|prev| prev.name == spec.name) {
            panic!("Duplicate parameter '{}'.", spec.name);
        }

        let mut outputs = self.outputs;
        outputs.push(spec);
        Self {
            operator_id: self.operator_id,
            description: self.description,
            inputs: self.inputs,
            outputs,
        }
    }

    /// Validates the provided input types against the specification
    pub fn validate_inputs(
        &self,
        input_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        self.validate_parameters("input", &self.inputs, input_types)
    }

    /// Validates the provided output types against the specification
    pub fn validate_outputs(
        &self,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        self.validate_parameters("output", &self.outputs, output_types)
    }

    /// Validates both inputs and outputs
    pub fn validate(
        &self,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        self.validate_inputs(input_types)?;
        self.validate_outputs(output_types)?;
        Ok(())
    }

    fn validate_parameters(
        &self,
        param_type: &str,
        specs: &[ParameterSpec],
        provided: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        // Check for required parameters
        let required_params: Vec<_> = specs
            .iter()
            .filter(|spec| spec.arity == ParameterArity::Required)
            .collect();

        for spec in &required_params {
            if !provided.contains_key(&spec.name) {
                return Err(format!(
                    "Missing required {} parameter '{}' of type {:?}",
                    param_type, spec.name, spec.data_type
                ));
            }

            if provided[&spec.name].type_name != spec.data_type.type_name {
                return Err(format!(
                    "{} parameter '{}' expected type {:?}, but got {:?}",
                    param_type, spec.name, spec.data_type, provided[&spec.name]
                ));
            }
        }

        // Check for unknown parameters
        let expected_names: BTreeMap<String, &DataTypeDescription> = specs
            .iter()
            .map(|spec| (spec.name.clone(), &spec.data_type))
            .collect();

        for (name, data_type) in provided {
            match expected_names.get(name) {
                Some(expected_type) => {
                    if data_type.type_name != expected_type.type_name {
                        return Err(format!(
                            "{param_type} parameter '{name}' expected type {expected_type:?}, but got {data_type:?}"
                        ));
                    }
                }
                None => {
                    return Err(format!(
                        "Unexpected {} parameter '{}'. Expected parameters: [{}]",
                        param_type,
                        name,
                        specs
                            .iter()
                            .map(|s| s.name.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    ));
                }
            }
        }

        Ok(())
    }
}
type BuildInputRefMap<'a> = BTreeMap<&'a str, Option<&'a dyn Any>>;
type BuildOutputArcMap<'a> = BTreeMap<String, Option<AnyArc>>;

/// Implementation of a `BuildPlan` operator.
pub trait ColumnBuildOperator: 'static {
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
    operator: Box<dyn ColumnBuildOperator>,
}

impl Debug for ColumnBuilder {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if !f.alternate() {
            f.debug_struct("ColumnBuilder")
                .field("id", &self.build_plan.operator_id)
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
    /// * `env` - An environment that can create the operator based on the build plan.
    ///
    /// # Returns
    ///
    /// A result containing a `BoundPlanBuilder` if successful, or an error message if the binding fails.
    #[must_use]
    pub fn new_for_plan<E>(
        table_schema: &TableSchema,
        build_plan: &BuildPlan,
        env: &E,
    ) -> Result<ColumnBuilder, String>
    where
        E: OpEnvironment,
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

        let context = ColumnBuildOperationInitContext::new(
            build_plan.clone(),
            input_types.clone(),
            output_types.clone(),
        );
        let operator = env.build(&context)?;

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
    ) -> Result<Box<dyn ColumnBuildOperator>, String>;
}

/// A factory for building operators within a specific namespace.
#[derive(Debug)]
pub struct MapOperatorFactory {
    /// The operator factory to be used for building operators.
    pub operations: BTreeMap<String, Arc<dyn BuildOperatorFactory>>,
}

impl Default for MapOperatorFactory {
    fn default() -> Self {
        Self::new()
    }
}

impl MapOperatorFactory {
    /// Create a new `MapOperatorFactory`.
    pub fn new() -> Self {
        MapOperatorFactory {
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
            panic!("Operation '{operation}' is already registered in the factory");
        }
        self.operations.insert(operation.to_string(), factory);
    }
}

impl BuildOperatorFactory for MapOperatorFactory {
    fn init_operator(
        &self,
        build_plan: &BuildPlan,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<Box<dyn ColumnBuildOperator>, String> {
        if let Some(factory) = self.operations.get(&build_plan.operator_id) {
            factory.init_operator(build_plan, input_types, output_types)
        } else {
            Err(format!(
                "No operator factory registered for operation '{}' in namespace '{}'",
                build_plan.operator_id, build_plan.operator_id
            ))
        }
    }
}

/// Binding for an operator that can be initialized with a build plan.
pub trait ColumnBuildOperatorBuilder {
    /// Returns the operator ID.
    fn operator_id(&self) -> &String {
        self.spec()
            .operator_id
            .as_ref()
            .expect("Spec must have an operator id")
    }

    /// Returns the operator specification.
    fn spec(&self) -> &OperatorSpec;

    /// Validates a build plan against the input and output types using an `OpInitContext`.
    ///
    /// # Arguments
    ///
    /// * `context` - The context containing the build plan and input/output types.
    ///
    /// # Returns
    ///
    /// A `Result<(), String>` where:
    /// * `Ok` indicates successful validation,
    /// * `Err` contains an error message if validation fails.
    fn validate(
        &self,
        context: &ColumnBuildOperationInitContext,
    ) -> Result<(), String> {
        self.build(context).map(|_| ())
    }

    /// Inits a build plan against the input and output types using an `OpInitContext`.
    ///
    /// # Arguments
    ///
    /// * `context` - The context containing the build plan and input/output types.
    ///
    /// # Returns
    ///
    /// A `Result<Box<dyn BuildOperator>, String>` where:
    /// * `Ok` contains a boxed operator that implements the `BuildOperator` trait,
    /// * `Err` contains an error message if the initialization fails.
    fn build(
        &self,
        context: &ColumnBuildOperationInitContext,
    ) -> Result<Box<dyn ColumnBuildOperator>, String>;
}

/// Context for validating and initializing a column build operation.
#[derive(Debug, Clone)]
pub struct ColumnBuildOperationInitContext {
    /// The operator specification for the operator being initialized.
    build_plan: BuildPlan,

    /// A map of input parameter names to their data types.
    input_types: BTreeMap<String, DataTypeDescription>,

    /// A map of output parameter names to their data types.
    output_types: BTreeMap<String, DataTypeDescription>,
}

impl ColumnBuildOperationInitContext {
    /// Creates a new `ColumnBuildOperationInitContext` with the specified build plan and input/output types.
    pub fn new(
        build_plan: BuildPlan,
        input_types: BTreeMap<String, DataTypeDescription>,
        output_types: BTreeMap<String, DataTypeDescription>,
    ) -> Self {
        Self {
            build_plan,
            input_types,
            output_types,
        }
    }

    /// Returns a reference to the build plan.
    pub fn build_plan(&self) -> &BuildPlan {
        &self.build_plan
    }

    /// Returns a reference to the input types.
    pub fn input_types(&self) -> &BTreeMap<String, DataTypeDescription> {
        &self.input_types
    }

    /// Returns a reference to the output types.
    pub fn output_types(&self) -> &BTreeMap<String, DataTypeDescription> {
        &self.output_types
    }
}

/// JsonConfigOpBinding is a concrete implementation of OpBinding that uses a JSON configuration to initialize an operator.
pub struct JsonConfigOpBinding<T>
where
    T: DeserializeOwned + ColumnBuildOperator,
{
    spec: OperatorSpec,
    phantom_data: PhantomData<T>,
}

impl<T> JsonConfigOpBinding<T>
where
    T: DeserializeOwned + ColumnBuildOperator,
{
    /// Creates a new `SpecConfigOpBinding` with the given operator specification.
    pub fn new(spec: OperatorSpec) -> Self {
        if spec.operator_id.is_none() {
            panic!("OperatorSpec must have an operator_id");
        }
        Self {
            spec,
            phantom_data: PhantomData,
        }
    }
}

impl<T> ColumnBuildOperatorBuilder for JsonConfigOpBinding<T>
where
    T: DeserializeOwned + ColumnBuildOperator,
{
    fn spec(&self) -> &OperatorSpec {
        &self.spec
    }

    fn validate(&self, context: &ColumnBuildOperationInitContext) -> Result<(), String> {
        self.build(context).map(|_| ())
    }

    fn build(
        &self,
        context: &ColumnBuildOperationInitContext,
    ) -> Result<Box<dyn ColumnBuildOperator>, String> {
        self.spec.validate(context.input_types(), context.output_types())?;

        let config = &context.build_plan().config;
        let loader: Box<T> = Box::new(serde_json::from_value(config.clone()).map_err(
            |_| {
                format!(
                    "Invalid config: {}",
                    serde_json::to_string_pretty(config).unwrap()
                )
            },
        )?);

        Ok(loader)
    }
}

/// OpEnvironment is a trait that provides access to a collection of operator bindings.
pub trait OpEnvironment {
    /// Returns a reference to the map of operator bindings.
    fn operators(&self) -> &BTreeMap<String, Arc<dyn ColumnBuildOperatorBuilder>>;

    /// Validates a build plan against the input and output types.
    ///
    /// # Arguments
    ///
    /// * `build_plan` - The build plan for the operator.
    /// * `input_types` - A map of input parameter names to their data types.
    /// * `output_types` - A map of output parameter names to their data types.
    ///
    /// # Returns
    ///
    /// A `Result<Box<dyn BuildOperator>, String>` where:
    /// * `Ok` contains a boxed operator that implements the `BuildOperator` trait,
    /// * `Err` contains an error message if the initialization fails.
    fn lookup_operation_builder(
        &self,
        operator_id: &str,
    ) -> Result<Arc<dyn ColumnBuildOperatorBuilder>, String> {
        Ok(self
            .operators()
            .get(operator_id)
            .ok_or_else(|| format!("Operator '{operator_id}' not found in environment."))?
            .clone())
    }

    /// Validates the operator against the build plan and input/output types.
    fn validate(
        &self,
        context: &ColumnBuildOperationInitContext,
    ) -> Result<(), String> {
        self.lookup_operation_builder(&context.build_plan().operator_id)?
            .validate(context)
    }

    /// Builds an operator based on the provided context.
    fn build(
        &self,
        context: &ColumnBuildOperationInitContext,
    ) -> Result<Box<dyn ColumnBuildOperator>, String> {
        let builder = self.lookup_operation_builder(&context.build_plan().operator_id)?;
        builder.validate(context)?;
        builder.build(context)
    }

    /// Extends a schema with a new operation.
    ///
    /// Plans new output columns and adds a build plan for the operation.
    ///
    /// Validates the inputs, outputs, and configs against the environment.
    ///
    /// # Arguments
    ///
    /// * `schema` - A mutable reference to the `TableSchema` to be extended.
    /// * `call` - A `CallBuilder` that describes the operation to be added.
    ///
    /// # Returns
    ///
    /// A `Result<BuildPlan, String>` where:
    /// * `Ok` contains the build plan for the operation,
    /// * `Err` contains an error message if the operation fails.
    fn plan_operation(
        &self,
        schema: &mut TableSchema,
        call: CallBuilder,
    ) -> Result<BuildPlan, String> {
        let operator_id = &call.operator_id;

        let input_types: BTreeMap<String, DataTypeDescription> = call
            .inputs
            .iter()
            .map(|(pname, cname)| (pname.to_string(), schema[cname.as_str()].data_type.clone()))
            .collect();

        let binding = self.lookup_operation_builder(operator_id)?;
        let spec = binding.spec();

        let input_bindings: Vec<(&str, &str)> = call
            .inputs
            .iter()
            .map(|(pname, cname)| (pname.as_str(), cname.as_str()))
            .collect();

        let output_bindings: Vec<(&str, &str)> = call
            .outputs
            .iter()
            .map(|(pname, cname)| (pname.as_str(), cname.as_str()))
            .collect();

        let mut plan = BuildPlan::for_operator(operator_id)
            .with_inputs(&input_bindings)
            .with_outputs(&output_bindings);

        if let Some(description) = &spec.description {
            plan = plan.with_description(description);
        }
        if let Some(config) = &call.config {
            plan = plan.with_config(config);
        }

        // Fuse static output types with the call's output extensions.
        let output_plan = spec
            .output_plan()
            .iter()
            .map(|(pname, dtype, description)| {
                let mut dtype = dtype.clone();
                if let Some(extension) = call.output_extensions.get(pname) {
                    dtype.extension = extension.clone();
                }

                (pname.clone(), dtype, description.clone())
            })
            .collect::<Vec<_>>();

        let context = ColumnBuildOperationInitContext::new(
            plan.clone(),
            input_types.clone(),
            output_plan
                .iter()
                .map(|(pname, dtype, _)| (pname.clone(), dtype.clone()))
                .collect(),
        );

        self.validate(&context)?;

        schema.add_build_plan_and_outputs(plan.clone(), &output_plan)?;

        Ok(plan)
    }
}

/// A builder for constructing a call to an operator in a `BuildPlan`.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CallBuilder {
    /// The ID of the operator to be called.
    pub operator_id: String,

    /// A map from formal parameter names to column names for inputs.
    pub inputs: BTreeMap<String, String>,

    /// A map from formal parameter names to column names for outputs.
    pub outputs: BTreeMap<String, String>,

    /// A map from formal parameter names to their extensions, serialized as JSON.
    pub output_extensions: BTreeMap<String, serde_json::Value>,

    /// An optional configuration for the call, serialized as JSON.
    pub config: Option<serde_json::Value>,
}

impl CallBuilder {
    /// Creates a new `CallBuilder` for the specified operator ID.
    ///
    /// # Arguments
    ///
    /// * `operator_id` - The ID of the operator to be called.
    pub fn new(operator_id: &str) -> Self {
        Self {
            operator_id: operator_id.to_string(),
            inputs: BTreeMap::new(),
            outputs: BTreeMap::new(),
            output_extensions: BTreeMap::new(),
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

    /// Adds an output extension to the call builder.
    pub fn with_output_extension<T>(
        mut self,
        pname: &str,
        extension: T,
    ) -> Self
    where
        T: Serialize,
    {
        if !self.outputs.contains_key(pname) {
            panic!("Output parameter '{pname}' must be defined before adding an extension.");
        }
        if self.output_extensions.contains_key(pname) {
            panic!("Output extension for '{pname}' already exists.");
        }
        self.output_extensions.insert(
            pname.to_string(),
            serde_json::to_value(extension).expect("Failed to serialize output extension"),
        );
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
}
/// MapOpEnvironment is a simple implementation of OpEnvironment that uses a BTreeMap to store operators.
pub struct MapOpEnvironment {
    operators: BTreeMap<String, Arc<dyn ColumnBuildOperatorBuilder>>,
}

impl Default for MapOpEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl MapOpEnvironment {
    /// Creates a new MapOpEnvironment with an empty operator map.
    pub fn new() -> Self {
        Self {
            operators: BTreeMap::new(),
        }
    }

    /// Creates a MapOpEnvironment from a list of operator bindings.
    ///
    /// # Arguments
    ///
    /// * `bindings` - A slice of Arc-wrapped operator bindings to initialize the environment with.
    pub fn from_bindings(bindings: &[Arc<dyn ColumnBuildOperatorBuilder>]) -> Result<Self, String> {
        let mut this = Self::new();
        this.add_bindings(bindings)?;
        Ok(this)
    }

    /// Adds a new binding to the environment.
    ///
    /// # Arguments
    ///
    /// * `op` - An Arc-wrapped operator binding to add to the environment.
    pub fn add_binding(
        &mut self,
        binding: Arc<dyn ColumnBuildOperatorBuilder>,
    ) -> Result<(), String> {
        let id = binding.operator_id();
        if self.operators.contains_key(id) {
            return Err(format!(
                "Operator with ID '{id}' already exists in MapOpEnvironment."
            ));
        }
        self.operators.insert(id.clone(), binding);
        Ok(())
    }

    /// Adds multiple bindings to the environment.
    ///
    /// # Arguments
    ///
    /// * `bindings` - An iterable collection of Arc-wrapped operator bindings to add to the environment.
    ///
    /// # Returns
    ///
    /// A `Result<(), String>` indicating success or an error message if any binding fails to be added.
    pub fn add_bindings<'a, B>(
        &mut self,
        bindings: B,
    ) -> Result<(), String>
    where
        B: IntoIterator<Item = &'a Arc<dyn ColumnBuildOperatorBuilder>>,
    {
        for binding in bindings.into_iter() {
            self.add_binding(binding.clone())?;
        }
        Ok(())
    }
}

impl OpEnvironment for MapOpEnvironment {
    fn operators(&self) -> &BTreeMap<String, Arc<dyn ColumnBuildOperatorBuilder>> {
        &self.operators
    }
}

/// UnionEnvironment combines multiple OpEnvironment instances into a single environment.
pub struct UnionEnvironment {
    environments: Vec<Arc<dyn OpEnvironment>>,
    operators: BTreeMap<String, Arc<dyn ColumnBuildOperatorBuilder>>,
}

impl Default for UnionEnvironment {
    fn default() -> Self {
        Self::new()
    }
}

impl UnionEnvironment {
    /// Creates a new UnionEnvironment with no environments.
    pub fn new() -> Self {
        Self {
            environments: Vec::new(),
            operators: BTreeMap::new(),
        }
    }

    /// Creates a UnionEnvironment from a list of OpEnvironment instances.
    ///
    /// # Arguments
    ///
    /// * `environments` - A slice of Arc-wrapped OpEnvironment instances.
    ///
    /// # Panics
    ///
    /// If any operator names are duplicated across the environments.
    pub fn from_list(environments: &[Arc<dyn OpEnvironment>]) -> Self {
        let mut env = Self::new();
        for e in environments {
            env.add_environment(e.clone());
        }
        env
    }

    /// Adds a new OpEnvironment to the UnionEnvironment.
    ///
    /// # Arguments
    ///
    /// * `env` - An Arc-wrapped OpEnvironment instance to add.
    ///
    /// # Panics
    ///
    /// If the new environment contains operators that already exist in any of the previously added environments.
    pub fn add_environment(
        &mut self,
        env: Arc<dyn OpEnvironment>,
    ) {
        // Ensure no duplicates
        let new_keys: HashSet<&str> = env.operators().keys().map(|s| s.as_str()).collect();

        for existing_env in &self.environments {
            let existing_keys = existing_env.operators().keys();

            for key in existing_keys {
                if new_keys.contains(key.as_str()) {
                    panic!("Duplicate operator '{key}' found in UnionEnvironment.");
                }
            }
        }

        self.operators.extend(env.operators().clone());
        self.environments.push(env);
    }
}

impl OpEnvironment for UnionEnvironment {
    fn operators(&self) -> &BTreeMap<String, Arc<dyn ColumnBuildOperatorBuilder>> {
        &self.operators
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::{
        AnyArc, BuildPlan, ColumnSchema, DataTypeDescription, RowBatch, TableSchema,
    };

    use crate::core::operators::{
        JsonConfigOpBinding, MapOpEnvironment, OperatorSpec, ParameterSpec,
    };
    use crate::define_operator_id;
    use indoc::indoc;
    use serde::{Deserialize, Serialize};
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::fmt::Debug;
    use std::sync::Arc;

    define_operator_id!(ADD);

    #[derive(Debug, Serialize, Deserialize)]
    struct AddOperator {
        bias: i32,
    }

    fn add_operator_op_binding() -> Arc<JsonConfigOpBinding<AddOperator>> {
        Arc::new(JsonConfigOpBinding::new(
            OperatorSpec::new()
                .with_operator_id(ADD)
                .with_description("Adds inputs with a bias")
                .with_input(ParameterSpec::new::<i32>("x").with_description("First input"))
                .with_input(ParameterSpec::new::<i32>("y").with_description("Second input"))
                .with_output(
                    ParameterSpec::new::<i32>("result")
                        .with_description("Result of addition with bias"),
                ),
        ))
    }

    impl ColumnBuildOperator for AddOperator {
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
    #[should_panic(expected = "'x' expected type")]
    fn test_bad_input_data_type() {
        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<String>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<i32>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env = MapOpEnvironment::from_bindings(&[add_operator_op_binding()]).unwrap();

        let _builder = ColumnBuilder::new_for_plan(&schema, &schema.build_plans[0], &env).unwrap();
    }

    #[test]
    #[should_panic(expected = "'result' expected type")]
    fn test_bad_output_data_type() {
        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<i32>("a").with_description("First input"),
            ColumnSchema::new::<i32>("b").with_description("Second input"),
            ColumnSchema::new::<String>("c").with_description("Output"),
        ]);

        schema
            .add_build_plan(
                BuildPlan::for_operator(ADD)
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env = MapOpEnvironment::from_bindings(&[add_operator_op_binding()]).unwrap();

        let _builder = ColumnBuilder::new_for_plan(&schema, &schema.build_plans[0], &env).unwrap();
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
                BuildPlan::for_operator(ADD)
                    .with_description("Adds inputs with a bias")
                    .with_config(AddOperator { bias: 10 })
                    .with_inputs(&[("x", "a"), ("y", "b")])
                    .with_outputs(&[("result", "c")]),
            )
            .unwrap();

        let env = MapOpEnvironment::from_bindings(&[add_operator_op_binding()]).unwrap();

        let builder = ColumnBuilder::new_for_plan(&schema, &schema.build_plans[0], &env).unwrap();

        assert_eq!(
            format!("{builder:#?}"),
            indoc! {r#"
               ColumnBuilder {
                   build_plan: BuildPlan {
                       operator_id: "bimm_firehose::core::operators::tests::ADD",
                       description: Some(
                           "Adds inputs with a bias",
                       ),
                       config: Object {
                           "bias": Number(10),
                       },
                       inputs: {
                           "x": "a",
                           "y": "b",
                       },
                       outputs: {
                           "result": "c",
                       },
                   },
               }"#,
            }
        );

        assert_eq!(builder.effective_batch_size(), 1);

        assert_eq!(builder.build_plan.operator_id, ADD);

        let mut batch = RowBatch::with_size(Arc::new(schema.clone()), 2);
        batch[0].set_columns(&schema, &["a", "b"], [Arc::new(10), Arc::new(20)]);
        batch[1].set_columns(&schema, &["a", "b"], [Arc::new(-5), Arc::new(2)]);

        builder.apply_batch(batch.rows.as_mut_slice()).unwrap();

        assert_eq!(batch[0].get_column::<i32>(&schema, "c").unwrap(), &40);
        assert_eq!(batch[1].get_column::<i32>(&schema, "c").unwrap(), &7);
    }

    #[test]
    fn test_operator_spec_validation() {
        let spec = OperatorSpec::new()
            .with_input(ParameterSpec::new::<i32>("input1"))
            .with_input(ParameterSpec::new::<String>("input2").with_arity(ParameterArity::Optional))
            .with_output(ParameterSpec::new::<f64>("output"));

        let mut input_types = BTreeMap::new();
        input_types.insert("input1".to_string(), DataTypeDescription::new::<i32>());
        input_types.insert("input2".to_string(), DataTypeDescription::new::<String>());

        let mut output_types = BTreeMap::new();
        output_types.insert("output".to_string(), DataTypeDescription::new::<f64>());

        assert!(spec.validate(&input_types, &output_types).is_ok());
    }

    #[test]
    fn test_path_ident() {
        define_operator_id!(FOO);

        assert_eq!(FOO, concat!(module_path!(), "::FOO"));
    }

    #[test]
    fn test_map_op_environment() {
        let _env = MapOpEnvironment::new();
    }
}
