use crate::core::operations::environment::MapOpEnvironment;
use crate::core::operations::factory::FirehoseOperatorFactory;
use crate::core::operations::registration::FirehoseOperatorFactoryRegistration;
use crate::ops::image::tensor_loader::img_to_tensor_op_binding;
use burn::prelude::Backend;
use std::sync::Arc;

/// Image operators.
pub mod image;

/// Build the default environment.
///
/// This constructs a `MapOpEnvironment` and adds all operator builders
/// registered with `bimm_firehose::register_default_operator_builder!`.
///
/// Each call `build_default_environment` will create a new mutable environment.
pub fn init_default_operator_environment() -> MapOpEnvironment {
    let mut env = MapOpEnvironment::default();

    for reg in FirehoseOperatorFactoryRegistration::list_default_registrations() {
        env.add_operator(reg.get_builder()).unwrap();
    }

    env
}

/// Initialize all the built-in Burn device operators.
///
/// This function returns a vector of operator factories that can be used to create
/// operators for the specified Burn device backend.
pub fn init_burn_device_operators<B: Backend>(
    device: &B::Device
) -> Vec<Arc<dyn FirehoseOperatorFactory>> {
    vec![img_to_tensor_op_binding::<B>(device)]
}

/// Initialize a default + burn device operator environment.
pub fn init_burn_device_operator_environment<B: Backend>(device: &B::Device) -> MapOpEnvironment {
    let mut env = init_default_operator_environment();
    env.add_all_operators(init_burn_device_operators::<B>(device))
        .expect("Failed to add operators");
    env
}
