use crate::core::operations::environment::MapOpEnvironment;
use crate::core::operations::registration::FirehoseOperatorFactoryRegistration;

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
