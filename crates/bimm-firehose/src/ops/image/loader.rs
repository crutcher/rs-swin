use crate::core::op_spec::{OperatorSpec, ParameterSpec};
use crate::core::{BuildOperator, BuildOperatorFactory, BuildPlan, DataTypeDescription};
use crate::define_operator_id;
use crate::ops::image::{ImageShape, color_util};
use image::imageops::FilterType;
use image::{ColorType, DynamicImage, GenericImageView};
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Factory for creating an `ImageLoader` operator.
#[derive(Debug)]
pub struct ImageLoaderFactory {}

define_operator_id!(LOAD_IMAGE);

impl ImageLoaderFactory {
    /// Returns the operator specification for loading an image.
    pub fn load_image_op_spec() -> OperatorSpec {
        OperatorSpec::new()
            .with_operator_id(LOAD_IMAGE)
            .with_input(ParameterSpec::new::<String>("path"))
            .with_output(
                ParameterSpec::new::<DynamicImage>("image")
                    .with_description("Image loaded from disk."),
            )
            .with_description("Loads an image from disk.")
    }
}

impl BuildOperatorFactory for ImageLoaderFactory {
    fn init_operator(
        &self,
        build_plan: &BuildPlan,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<Box<dyn BuildOperator>, String> {
        Self::load_image_op_spec().validate(input_types, output_types)?;

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
        ColumnSchema, RowBatch, TableSchema, experimental_run_batch,
        extend_schema_with_operator_and_config,
    };
    use crate::ops::image::test_util;
    use crate::ops::image::test_util::assert_image_close;
    use image::DynamicImage;
    use std::sync::Arc;

    #[test]
    fn test_image_loader() -> Result<(), String> {
        let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");

        let image_path = temp_dir
            .path()
            .join("gradient.png")
            .to_string_lossy()
            .to_string();

        let source_image: DynamicImage = test_util::generate_gradient_pattern(ImageShape {
            width: 32,
            height: 32,
        })
        .into();

        source_image
            .save(&image_path)
            .expect("Failed to save test image");

        let mut schema = TableSchema::from_columns(&[ColumnSchema::new::<String>("path")]);

        extend_schema_with_operator_and_config(
            &mut schema,
            &ImageLoaderFactory::load_image_op_spec(),
            &[("path", "path")],
            &[("image", "image")],
            ImageLoader::default(),
        )?;

        extend_schema_with_operator_and_config(
            &mut schema,
            &ImageLoaderFactory::load_image_op_spec(),
            &[("path", "path")],
            &[("image", "resized_gray")],
            ImageLoader::default()
                .with_resize(
                    ResizeSpec::new(ImageShape {
                        width: 16,
                        height: 16,
                    })
                    .with_filter(FilterType::Nearest),
                )
                .with_recolor(ColorType::L16),
        )?;

        let schema = Arc::new(schema);

        let mut batch = RowBatch::with_size(schema.clone(), 1);
        batch[0].set_column(&schema, "path", Some(Arc::new(image_path)));

        let factory = ImageLoaderFactory {};
        experimental_run_batch(&mut batch, &factory)?;

        let loaded_image = batch[0]
            .get_column::<DynamicImage>(&schema, "image")
            .expect("Failed to get loaded image");
        assert_eq!(loaded_image.width(), 32);
        assert_eq!(loaded_image.height(), 32);
        assert_eq!(loaded_image.color(), ColorType::Rgb8);
        assert_image_close(loaded_image, &source_image, None);

        let gray_image = batch[0]
            .get_column::<DynamicImage>(&schema, "resized_gray")
            .expect("Failed to get resized gray image");
        assert_eq!(gray_image.width(), 16);
        assert_eq!(gray_image.height(), 16);
        assert_eq!(gray_image.color(), ColorType::L16);
        assert_image_close(
            gray_image,
            &source_image
                .resize(16, 16, FilterType::Nearest)
                .to_luma8()
                .into(),
            None,
        );

        Ok(())
    }
}
