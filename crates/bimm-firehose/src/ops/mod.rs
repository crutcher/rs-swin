use crate::core::MapOpEnvironment;
use crate::ops::image::loader::load_image_op_binding;

/// Image operators.
pub mod image;

/// Common environment for operators, including image loading.
pub fn common_environment() -> MapOpEnvironment {
    let mut env = MapOpEnvironment::new();

    env.add_binding(load_image_op_binding()).unwrap();

    env
}
