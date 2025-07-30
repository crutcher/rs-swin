use crate::core::operations::operator::FirehoseOperator;
use crate::core::operations::signature::FirehoseOperatorSignature;
use crate::core::schema::{BuildPlan, DataTypeDescription, FirehoseTableSchema};
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use std::marker::PhantomData;

/// A factory for creating `FirehoseOperator` instances from a specification.
pub trait FirehoseOperatorFactory: Debug + Send + Sync {
    /// Returns the operator ID.
    fn operator_id(&self) -> &String {
        self.signature()
            .operator_id
            .as_ref()
            .expect("Spec must have an operator id")
    }

    /// Returns the operator specification.
    fn signature(&self) -> &FirehoseOperatorSignature;

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
    fn init(
        &self,
        context: &dyn FirehoseOperatorInitContext,
    ) -> Result<Box<dyn FirehoseOperator>, String>;
}

/// The init interface for `FirehoseOperatorFactory`.
pub trait FirehoseOperatorInitContext {
    /// Returns the operator ID for the operator being initialized.
    fn operator_id(&self) -> &str;

    /// Returns the table schema for the operator being initialized.
    fn table_schema(&self) -> &FirehoseTableSchema;

    /// Returns a reference to the build plan.
    fn build_plan(&self) -> &BuildPlan;

    /// The operator signature for the operator being initialized.
    fn signature(&self) -> &FirehoseOperatorSignature;

    /// Returns the input type for a scalar input parameter.
    fn scalar_input_type(
        &self,
        param_name: &str,
    ) -> Option<DataTypeDescription> {
        let parameter = self.signature().get_input_parameter(param_name)?;
        assert!(parameter.arity.is_scalar(), "Parameter must be scalar");

        let col_name = self.build_plan().inputs.get(param_name)?;
        self.table_schema()
            .get_column(col_name)?
            .data_type
            .clone()
            .into()
    }

    /// Returns the output type for a scalar output parameter.
    fn scalar_output_type(
        &self,
        param_name: &str,
    ) -> Option<DataTypeDescription> {
        let parameter = self.signature().get_output_parameter(param_name)?;
        assert!(parameter.arity.is_scalar(), "Parameter must be scalar");

        let col_name = self.build_plan().outputs.get(param_name)?;
        self.table_schema()
            .get_column(col_name)?
            .data_type
            .clone()
            .into()
    }
}

/// A simple operator factory for types implementing `DeserializeOwned` and `FirehoseOperator`.
#[derive(Debug)]
pub struct SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    signature: FirehoseOperatorSignature,
    phantom_data: PhantomData<T>,
}

impl<T> SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    /// Creates a new `SpecConfigOpBinding` with the given operator specification.
    pub fn new(spec: FirehoseOperatorSignature) -> Self {
        if spec.operator_id.is_none() {
            panic!("OperatorSpec must have an operator_id");
        }
        Self {
            signature: spec,
            phantom_data: PhantomData,
        }
    }
}

impl<T> FirehoseOperatorFactory for SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    fn signature(&self) -> &FirehoseOperatorSignature {
        &self.signature
    }

    fn init(
        &self,
        context: &dyn FirehoseOperatorInitContext,
    ) -> Result<Box<dyn FirehoseOperator>, String> {
        let config = &context.build_plan().config;
        let loader: Box<T> = Box::new(serde_json::from_value(config.clone()).map_err(|_| {
            format!(
                "Invalid config: {}",
                serde_json::to_string_pretty(config).unwrap()
            )
        })?);

        Ok(loader)
    }
}

#[cfg(test)]
mod tests {
    use crate::core::operations::factory::SimpleConfigOperatorFactory;
    use crate::core::operations::operator::FirehoseOperator;
    use crate::core::operations::signature::{FirehoseOperatorSignature, ParameterSpec};
    use crate::define_firehose_operator_id;
    use serde::{Deserialize, Serialize};

    define_firehose_operator_id!(TEST_OP);

    #[derive(Debug, Serialize, Deserialize)]
    struct TestOperator {
        pub value: i32,
    }

    impl FirehoseOperator for TestOperator {}

    #[test]
    fn test_simple_config_operator_factory() {
        let signature = FirehoseOperatorSignature::from_operator_id(TEST_OP)
            .with_input(ParameterSpec::new::<i32>("input"))
            .with_output(ParameterSpec::new::<i32>("output"));

        let _factory = SimpleConfigOperatorFactory::<TestOperator>::new(signature);
    }
}
