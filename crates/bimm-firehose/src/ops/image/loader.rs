use crate::core::{BuildOperator, BuildOperatorFactory, BuildPlan, DataTypeDescription};
use image::imageops::FilterType;
use image::{ColorType, DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Factory for creating an `ImageLoader` operator.
#[derive(Debug)]
pub struct ImageLoaderFactory {}

impl BuildOperatorFactory for ImageLoaderFactory {
    fn init_operator(
        &self,
        build_plan: &BuildPlan,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<Box<dyn BuildOperator>, String> {
        let expected_type = DataTypeDescription::new::<DynamicImage>();

        if !input_types.contains_key("path") || input_types.len() != 1 {
            return Err(format!(
                "ImageLoader expects a single input 'path' of type {expected_type:?}, but got: {input_types:?}"
            ));
        }
        if !output_types.contains_key("image") || output_types.len() != 1 {
            return Err(format!(
                "ImageLoader expects a single output 'image' of type {expected_type:?}, but got: {output_types:?}"
            ));
        }
        if output_types["image"] != expected_type {
            return Err(format!(
                "ImageLoader expects output 'image' to be of type {:?}, but got: {:?}",
                expected_type, output_types["image"]
            ));
        }

        let factory: ImageLoader =
            serde_json::from_value(build_plan.config.clone()).expect("Invalid config");

        Ok(Box::new(factory))
    }
}

/// Represents the shape of an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageShape {
    /// The width of the image in pixels.
    pub width: u32,

    /// The height of the image in pixels.
    pub height: u32,
}

/// Represents the resize specification for an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ResizeSpec {
    /// The target shape of the image after resizing.
    pub shape: ImageShape,

    /// The filter type to use for resizing.
    pub filter: FilterType,
}

impl ResizeSpec {
    /// Creates a new `ResizeSpec` with the specified shape and filter.
    pub fn new(shape: ImageShape) -> Self {
        ResizeSpec {
            shape,
            filter: FilterType::CatmullRom,
        }
    }

    /// Extends the `ResizeSpec` with a new shape, keeping the existing filter type.
    pub fn with_filter(
        self,
        filter: FilterType,
    ) -> Self {
        ResizeSpec {
            shape: self.shape,
            filter,
        }
    }
}

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

fn convert_to_colortype(
    img: DynamicImage,
    target_type: ColorType,
) -> DynamicImage {
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
        _ => img, // fallback for unsupported types
    }
}

fn colortype_serializer<S>(
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

fn colortype_deserializer<'de, D>(deserializer: D) -> Result<Option<ColorType>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    let color: Option<TmpColorType> = Option::deserialize(deserializer)?;
    Ok(color.map(|c| c.into()))
}

/// An operator that loads an image from disk and optionally resizes it.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageLoader {
    /// If enabled, the loader will resize images to the specified shape.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub resize: Option<ResizeSpec>,

    /// If enabled, the loader will recolor images to the specified color type.
    #[serde(skip_serializing_if = "Option::is_none")]
    /// Upstream `image::ColorType` is not serializable in the latest release;
    /// but *is* in the base repo, and this will change.
    #[serde(
        serialize_with = "colortype_serializer",
        deserialize_with = "colortype_deserializer"
    )]
    #[serde(default)]
    pub recolor: Option<ColorType>,
}

impl Default for ImageLoader {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageLoader {
    /// Creates a new `ImageLoader` with optional resize and recolor specifications.
    pub fn new() -> Self {
        ImageLoader {
            resize: None,
            recolor: None,
        }
    }

    /// Extends the `ImageLoader` with a resize specification.
    ///
    /// ## Arguments
    ///
    /// * `resize`: The resize specification to apply to the image.
    ///
    /// ## Returns
    ///
    /// A new `ImageLoader` instance with the specified resize applied.
    pub fn with_resize(
        self,
        resize: ResizeSpec,
    ) -> Self {
        ImageLoader {
            resize: Some(resize),
            recolor: self.recolor,
        }
    }

    /// Extends the `ImageLoader` with a recolor specification.
    ///
    /// ## Arguments
    ///
    /// * `recolor`: The color type to convert the image to.
    ///
    /// ## Returns
    ///
    /// A new `ImageLoader` instance with the specified recolor applied.
    pub fn with_recolor(
        self,
        recolor: ColorType,
    ) -> Self {
        ImageLoader {
            resize: self.resize,
            recolor: Some(recolor),
        }
    }
}

impl BuildOperator for ImageLoader {
    fn apply(
        &self,
        inputs: &BTreeMap<&str, Option<&dyn Any>>,
    ) -> Result<BTreeMap<String, Option<Arc<dyn Any>>>, String> {
        let path = inputs
            .get("path")
            .and_then(|v| v.as_ref())
            .and_then(|v| v.downcast_ref::<String>())
            .ok_or("ImageLoader expects input 'path' to be a String")?;

        let mut image = image::open(path).map_err(|e| format!("Failed to load image: {e}"))?;

        if let Some(spec) = &self.resize {
            if image.dimensions() != (spec.shape.width, spec.shape.height) {
                image = image.resize(spec.shape.width, spec.shape.height, spec.filter);
            }
        }

        if let Some(color) = &self.recolor {
            let color: ColorType = *color;
            image = convert_to_colortype(image, color);
        }

        {
            let mut result: BTreeMap<String, Option<Arc<dyn Any>>> = BTreeMap::new();
            result.insert("image".to_string(), Some(Arc::new(image)));
            Ok(result)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::{
        BuildPlan, ColumnSchema, DataTypeDescription, RowBatch, TableSchema, experimental_run_batch,
    };
    use image::{DynamicImage, ImageBuffer, Rgb, RgbImage};
    use std::sync::Arc;

    fn generate_gradient_pattern() -> RgbImage {
        ImageBuffer::from_fn(32, 32, |x, y| {
            let r = ((x as f32 / 31.0) * 255.0) as u8;
            let g = ((y as f32 / 31.0) * 255.0) as u8;
            let b = (((x + y) as f32 / 62.0) * 255.0) as u8;

            Rgb([r, g, b])
        })
    }

    #[test]
    fn test_image_loader() {
        let temp_dir = tempfile::tempdir().unwrap();

        let image = generate_gradient_pattern();

        let image_path = temp_dir
            .path()
            .join("gradient.png")
            .to_string_lossy()
            .to_string();

        image.save(&image_path).expect("Failed to save test image");

        let mut schema = TableSchema::from_columns(&[ColumnSchema::new::<String>("path")]);
        schema
            .add_build_plan_and_outputs(
                BuildPlan::for_operator(("image", "load_image"))
                    .with_description("Loads image from disk")
                    .with_inputs(&[("path", "path")])
                    .with_outputs(&[("image", "image")])
                    .with_config(ImageLoader::new()),
                &[(
                    "image",
                    DataTypeDescription::new::<DynamicImage>(),
                    "Loaded image",
                )],
            )
            .expect("Failed to add build plan");
        schema
            .add_build_plan_and_outputs(
                BuildPlan::for_operator(("image", "load_image"))
                    .with_description("Loads image from disk")
                    .with_inputs(&[("path", "path")])
                    .with_outputs(&[("image", "resized_gray")])
                    .with_config(
                        ImageLoader::new()
                            .with_resize(ResizeSpec::new(ImageShape {
                                width: 16,
                                height: 16,
                            }))
                            .with_recolor(ColorType::L16),
                    ),
                &[(
                    "image",
                    DataTypeDescription::new::<DynamicImage>(),
                    "Loaded image",
                )],
            )
            .expect("Failed to add build plan");

        let schema = Arc::new(schema);

        let mut batch = RowBatch::with_size(schema.clone(), 1);
        batch[0].set_column(&schema, "path", Some(Arc::new(image_path)));

        let factory = ImageLoaderFactory {};
        experimental_run_batch(&mut batch, &factory).expect("Failed to complete batch");

        let loaded_image = batch[0]
            .get_column::<DynamicImage>(&schema, "image")
            .expect("Failed to get loaded image");
        assert_eq!(loaded_image.width(), 32);
        assert_eq!(loaded_image.height(), 32);
        assert_eq!(loaded_image.color(), ColorType::Rgb8);
        // TODO: compare pixel values

        let gray_image = batch[0]
            .get_column::<DynamicImage>(&schema, "resized_gray")
            .expect("Failed to get resized gray image");
        assert_eq!(gray_image.width(), 16);
        assert_eq!(gray_image.height(), 16);
        assert_eq!(gray_image.color(), ColorType::L16);
        // TODO: compare pixel values
    }
}
