use crate::core::operations::factory::{FirehoseOperatorFactory, OperatorInitializationContext};
use crate::core::operations::operator::FirehoseOperator;
use crate::core::operations::planner::OperationPlanner;
use crate::core::operations::registration;
use crate::core::schema::{BuildPlan, DataTypeDescription, TableSchema};
use std::collections::{BTreeMap, HashSet};
use std::sync::Arc;

/// Build the default environment.
///
/// This constructs a `MapOpEnvironment` and adds all operator builders
/// registered with `bimm_firehose::register_default_operator_builder!`.
///
/// Each call `build_default_environment` will create a new mutable environment.
pub fn new_default_operator_environment() -> MapOpEnvironment {
    let mut env = MapOpEnvironment::new();

    for reg in registration::FirehoseOperatorFactoryRegistration::list_default_registrations() {
        env.add_binding(reg.get_builder()).unwrap();
    }

    env
}

/// OpEnvironment is a trait that provides access to a collection of operator bindings.
pub trait OpEnvironment {
    /// Returns a reference to the map of operator bindings.
    fn operators(&self) -> &BTreeMap<String, Arc<dyn FirehoseOperatorFactory>>;

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
    ) -> Result<Arc<dyn FirehoseOperatorFactory>, String> {
        Ok(self
            .operators()
            .get(operator_id)
            .ok_or_else(|| format!("Operator '{operator_id}' not found in environment."))?
            .clone())
    }

    /// Validates the operator against the build plan and input/output types.
    fn validate(
        &self,
        context: &OperatorInitializationContext,
    ) -> Result<(), String> {
        let builder = self.lookup_operation_builder(&context.build_plan().operator_id)?;
        builder
            .spec()
            .validate(context.input_types(), context.output_types())?;
        builder.supplemental_validation(context)
    }

    /// Builds an operator based on the provided context.
    fn build(
        &self,
        context: &OperatorInitializationContext,
    ) -> Result<Box<dyn FirehoseOperator>, String> {
        let builder = self.lookup_operation_builder(&context.build_plan().operator_id)?;
        builder.supplemental_validation(context)?;
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
    /// * `table_schema` - A mutable reference to the `TableSchema` to be extended.
    /// * `call` - A `CallBuilder` that describes the operation to be added.
    ///
    /// # Returns
    ///
    /// A `Result<BuildPlan, String>` where:
    /// * `Ok` contains the build plan for the operation,
    /// * `Err` contains an error message if the operation fails.
    fn plan_operation(
        &self,
        table_schema: &mut TableSchema,
        call: OperationPlanner,
    ) -> Result<BuildPlan, String> {
        let operator_id = &call.operator_id;

        let input_types: BTreeMap<String, DataTypeDescription> = call
            .inputs
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.to_string(),
                    table_schema[cname.as_str()].data_type.clone(),
                )
            })
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

        let context = OperatorInitializationContext::new(
            table_schema.clone(),
            plan.clone(),
            input_types.clone(),
            output_plan
                .iter()
                .map(|(pname, dtype, _)| (pname.clone(), dtype.clone()))
                .collect(),
        );

        self.validate(&context)?;

        table_schema.add_build_plan_and_outputs(plan.clone(), &output_plan)?;

        Ok(plan)
    }
}

/// MapOpEnvironment is a simple implementation of OpEnvironment that uses a BTreeMap to store operators.
pub struct MapOpEnvironment {
    operators: BTreeMap<String, Arc<dyn FirehoseOperatorFactory>>,
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
    pub fn from_bindings(bindings: &[Arc<dyn FirehoseOperatorFactory>]) -> Result<Self, String> {
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
        binding: Arc<dyn FirehoseOperatorFactory>,
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
        B: IntoIterator<Item = &'a Arc<dyn FirehoseOperatorFactory>>,
    {
        for binding in bindings.into_iter() {
            self.add_binding(binding.clone())?;
        }
        Ok(())
    }
}

impl OpEnvironment for MapOpEnvironment {
    fn operators(&self) -> &BTreeMap<String, Arc<dyn FirehoseOperatorFactory>> {
        &self.operators
    }
}

/// UnionEnvironment combines multiple OpEnvironment instances into a single environment.
pub struct UnionEnvironment {
    environments: Vec<Arc<dyn OpEnvironment>>,
    operators: BTreeMap<String, Arc<dyn FirehoseOperatorFactory>>,
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
    fn operators(&self) -> &BTreeMap<String, Arc<dyn FirehoseOperatorFactory>> {
        &self.operators
    }
}
