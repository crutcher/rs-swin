use crate::core::operations::factory::FirehoseOperatorFactory;
use std::fmt::Debug;
use std::sync::Arc;

// This leverages the `inventory` crate to define a collection scheme for
// later invocations of `inventory::submit! { <instance> }` by the registration macro.
inventory::collect!(FirehoseOperatorFactoryRegistration);

/// Struct describing a name to constructor for an operator builder.
///
/// Used by `register_default_operator_builder!` to register operator builders
/// which do not require any additional configuration or parameters.
#[derive(Clone)]
pub struct FirehoseOperatorFactoryRegistration {
    /// The operator ID.
    pub operator_id: &'static str,

    /// The builder.
    pub supplier: fn() -> Arc<dyn FirehoseOperatorFactory>,
}

impl Debug for FirehoseOperatorFactoryRegistration {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("FirehoseOperatorFactoryRegistration")
            .field("operator_id", &self.operator_id)
            .finish()
    }
}

impl FirehoseOperatorFactoryRegistration {
    /// Creates a new registration.
    pub const fn new(
        operator_id: &'static str,
        supplier: fn() -> Arc<dyn FirehoseOperatorFactory>,
    ) -> Self {
        Self {
            operator_id,
            supplier,
        }
    }

    /// Returns the builder.
    pub fn get_builder(&self) -> Arc<dyn FirehoseOperatorFactory> {
        let builder = (self.supplier)();

        assert_eq!(
            builder.operator_id(),
            self.operator_id,
            "Builder operator ID does not match registration ID: {} != {}",
            builder.operator_id(),
            self.operator_id,
        );

        builder
    }

    /// List all default operator registrations.
    pub fn list_default_registrations() -> Vec<&'static FirehoseOperatorFactoryRegistration> {
        inventory::iter::<FirehoseOperatorFactoryRegistration>
            .into_iter()
            .collect()
    }
}
