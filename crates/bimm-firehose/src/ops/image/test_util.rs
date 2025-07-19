use image::{ImageBuffer, Rgb, RgbImage};
use crate::ops::image::ImageShape;

/// Generates a simple gradient pattern image.
pub fn generate_gradient_pattern(shape: ImageShape) -> RgbImage {
    let r_scale = shape.width as f32 - 1.0;
    let g_scale = shape.height as f32 - 1.0;
    let b_scale = (shape.width + shape.height - 2) as f32;
    
    ImageBuffer::from_fn(shape.width, shape.height, |x, y| {
        let a = x as f32 * 255.0;
        let b = y as f32 * 255.0;
        
        let r = (a / r_scale) as u8;
        let g = (b / g_scale) as u8;
        let b = ((a + b) / b_scale) as u8;

        Rgb([r, g, b])
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_gradient_pattern() {
        let shape = ImageShape { width: 32, height: 32 };
        let image = generate_gradient_pattern(shape);

        assert_eq!(image.width(), 32);
        assert_eq!(image.height(), 32);

        // Check some pixel values
        assert_eq!(image.get_pixel(0, 0), &Rgb([0, 0, 0]));
        assert_eq!(image.get_pixel(16, 16), &Rgb([131, 131, 131]));
        assert_eq!(image.get_pixel(16, 31), &Rgb([131, 255, 193]));
        assert_eq!(image.get_pixel(31, 31), &Rgb([255, 255, 255]));
    }
}