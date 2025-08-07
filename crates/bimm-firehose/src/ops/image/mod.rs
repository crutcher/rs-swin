use serde::{Deserialize, Serialize};

/// `image::ColorType` utils.
pub mod colortype_support;
/// Image loader operators.
pub mod loader;

/// Image augmentation operators.
pub mod augmentation;

/// Image/Tensor conversion utilities.
pub mod burn;
/// Image test utilities.
pub mod test_util;

/// Represents the shape of an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageShape {
    /// The width of the image in pixels.
    pub width: u32,

    /// The height of the image in pixels.
    pub height: u32,
}

pub use image::ColorType;
