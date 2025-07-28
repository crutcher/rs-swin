use crate::core::operations::operator::{FirehoseOperator, OperatorInitializationContext};
use crate::core::operations::signature::FirehoseOperatorSignature;
use serde::de::DeserializeOwned;
use std::fmt::Debug;
use std::marker::PhantomData;

/// An operator factory which deserializes the operator from a JSON value.
#[derive(Debug)]
pub struct SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    spec: FirehoseOperatorSignature,
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
            spec,
            phantom_data: PhantomData,
        }
    }
}

impl<T> FirehoseOperatorFactory for SimpleConfigOperatorFactory<T>
where
    T: DeserializeOwned + FirehoseOperator,
{
    fn spec(&self) -> &FirehoseOperatorSignature {
        &self.spec
    }

    fn supplemental_validation(
        &self,
        context: &OperatorInitializationContext,
    ) -> Result<(), String> {
        self.build(context).map(|_| ())
    }

    fn build(
        &self,
        context: &OperatorInitializationContext,
    ) -> Result<Box<dyn FirehoseOperator>, String> {
        self.spec
            .validate(context.input_types(), context.output_types())?;

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

/// Binding for an operator that can be initialized with a build plan.
pub trait FirehoseOperatorFactory: Debug + Send + Sync {
    /// Returns the operator ID.
    fn operator_id(&self) -> &String {
        self.spec()
            .operator_id
            .as_ref()
            .expect("Spec must have an operator id")
    }

    /// Returns the operator specification.
    fn spec(&self) -> &FirehoseOperatorSignature;

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
    fn supplemental_validation(
        &self,
        context: &OperatorInitializationContext,
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
        context: &OperatorInitializationContext,
    ) -> Result<Box<dyn FirehoseOperator>, String>;
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
