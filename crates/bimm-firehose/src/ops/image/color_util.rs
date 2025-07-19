use image::{ColorType, DynamicImage};
use serde::{Deserialize, Serialize};

/// Temporary color type (with support for serialization and deserialization).
/// To be replaced with `image::ColorType` on the next release.
#[derive(Copy, PartialEq, Eq, Debug, Clone, Hash, Serialize, Deserialize)]
pub enum TmpColorType {
    /// Pixel is 8-bit luminance
    L8,
    /// Pixel is 8-bit luminance with an alpha channel
    La8,
    /// Pixel contains 8-bit R, G and B channels
    Rgb8,
    /// Pixel is 8-bit RGB with an alpha channel
    Rgba8,

    /// Pixel is 16-bit luminance
    L16,
    /// Pixel is 16-bit luminance with an alpha channel
    La16,
    /// Pixel is 16-bit RGB
    Rgb16,
    /// Pixel is 16-bit RGBA
    Rgba16,

    /// Pixel is 32-bit float RGB
    Rgb32F,
    /// Pixel is 32-bit float RGBA
    Rgba32F,
}

impl From<ColorType> for TmpColorType {
    fn from(color: ColorType) -> Self {
        match color {
            ColorType::L8 => TmpColorType::L8,
            ColorType::La8 => TmpColorType::La8,
            ColorType::Rgb8 => TmpColorType::Rgb8,
            ColorType::Rgba8 => TmpColorType::Rgba8,
            ColorType::L16 => TmpColorType::L16,
            ColorType::La16 => TmpColorType::La16,
            ColorType::Rgb16 => TmpColorType::Rgb16,
            ColorType::Rgba16 => TmpColorType::Rgba16,
            ColorType::Rgb32F => TmpColorType::Rgb32F,
            ColorType::Rgba32F => TmpColorType::Rgba32F,
            _ => panic!("Unsupported ColorType: {color:?}"),
        }
    }
}

impl From<TmpColorType> for ColorType {
    fn from(color: TmpColorType) -> Self {
        match color {
            TmpColorType::L8 => ColorType::L8,
            TmpColorType::La8 => ColorType::La8,
            TmpColorType::Rgb8 => ColorType::Rgb8,
            TmpColorType::Rgba8 => ColorType::Rgba8,
            TmpColorType::L16 => ColorType::L16,
            TmpColorType::La16 => ColorType::La16,
            TmpColorType::Rgb16 => ColorType::Rgb16,
            TmpColorType::Rgba16 => ColorType::Rgba16,
            TmpColorType::Rgb32F => ColorType::Rgb32F,
            TmpColorType::Rgba32F => ColorType::Rgba32F,
        }
    }
}

/// Convert an `image::DynamicImage` to a specific `ColorType`.
///
/// # Parameters
///
/// - `img`: The input image to convert.
/// - `target_type`: The target color type to convert the image to.
///
/// # Returns
///
/// A new `DynamicImage` with the specified color type.
pub fn convert_to_colortype(
    img: DynamicImage,
    target_type: ColorType,
) -> DynamicImage {
    if img.color() == target_type {
        return img;
    }
    match target_type {
        ColorType::L8 => DynamicImage::ImageLuma8(img.to_luma8()),
        ColorType::La8 => DynamicImage::ImageLumaA8(img.to_luma_alpha8()),
        ColorType::Rgb8 => DynamicImage::ImageRgb8(img.to_rgb8()),
        ColorType::Rgba8 => DynamicImage::ImageRgba8(img.to_rgba8()),
        ColorType::L16 => DynamicImage::ImageLuma16(img.to_luma16()),
        ColorType::La16 => DynamicImage::ImageLumaA16(img.to_luma_alpha16()),
        ColorType::Rgb16 => DynamicImage::ImageRgb16(img.to_rgb16()),
        ColorType::Rgba16 => DynamicImage::ImageRgba16(img.to_rgba16()),
        ColorType::Rgb32F => DynamicImage::ImageRgb32F(img.to_rgb32f()),
        ColorType::Rgba32F => DynamicImage::ImageRgba32F(img.to_rgba32f()),
        _ => panic!("Unsupported ColorType: {target_type:?}"),
    }
}

/// Adapter Serializer for `Option<ColorType>`.
pub fn option_colortype_serializer<S>(
    color: &Option<ColorType>,
    serializer: S,
) -> Result<S::Ok, S::Error>
where
    S: serde::Serializer,
{
    if color.is_none() {
        serializer.serialize_none()
    } else {
        let color = color.as_ref().unwrap();
        let color: Option<TmpColorType> = Some((*color).into());
        color.serialize(serializer)
    }
}

/// Adapter Deserializer for `Option<ColorType>`.
pub fn option_colortype_deserializer<'de, D>(deserializer: D) -> Result<Option<ColorType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let color: Option<TmpColorType> = Option::deserialize(deserializer)?;
    Ok(color.map(|c| c.into()))
}
