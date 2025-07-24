use serde::{Deserialize, Serialize};

/// `image::ColorType` utils.
pub mod color_util;
/// Image loader operators.
pub mod loader;
/// Image/Tensor conversion utilities.
pub mod tensor_loader;
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
