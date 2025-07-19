use crate::core::{BuildOperator, BuildOperatorFactory, BuildPlan, DataTypeDescription};
use image::imageops::FilterType;
use image::{ColorType, DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;
use crate::ops::image::{color_util, ImageShape};
use crate::ops::image::color_util::TmpColorType;

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
        serialize_with = "color_util::option_colortype_serializer",
        deserialize_with = "color_util::option_colortype_deserializer"
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
            image = color_util::convert_to_colortype(image, color);
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
        experimental_run_batch, BuildPlan, ColumnSchema, DataTypeDescription, RowBatch, TableSchema,
    };
    use image::DynamicImage;
    use std::sync::Arc;
    use crate::ops::image::test_util;

    #[test]
    fn test_image_loader() {
        let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
        
        let image_path = temp_dir
            .path()
            .join("gradient.png")
            .to_string_lossy()
            .to_string();

        let source_image = test_util::generate_gradient_pattern(
            ImageShape {
                width: 32,
                height: 32,
            }
        );

        source_image.save(&image_path).expect("Failed to save test image");

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
