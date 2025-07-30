use crate::core::operations::factory::SimpleConfigOperatorFactory;
use crate::core::operations::operator::{FirehoseOperator, OperatorRowTransaction};
use crate::core::operations::signature::{FirehoseOperatorSignature, ParameterSpec};
use crate::define_firehose_operator;
use crate::ops::image::{ImageShape, color_util};
use image::imageops::FilterType;
use image::{ColorType, DynamicImage};
use serde::{Deserialize, Serialize};
use std::sync::Arc;

define_firehose_operator!(
    LOAD_IMAGE,
    SimpleConfigOperatorFactory::<ImageLoader>::new(
        FirehoseOperatorSignature::from_operator_id(LOAD_IMAGE)
            .with_description("Loads an image from disk.")
            .with_input(
                ParameterSpec::new::<String>("path").with_description("Path to the image file."),
            )
            .with_output(
                ParameterSpec::new::<DynamicImage>("image")
                    .with_description("Image loaded from disk."),
            ),
    )
);

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

impl FirehoseOperator for ImageLoader {
    fn apply_row(
        &self,
        txn: &mut OperatorRowTransaction,
    ) -> Result<(), String> {
        let path = txn.get_required_scalar_input::<String>("path")?;

        let mut image = image::open(path).map_err(|e| format!("Failed to load image: {e}"))?;

        if let Some(spec) = &self.resize {
            if image.width() != spec.shape.width || image.height() != spec.shape.height {
                image = image.resize_exact(spec.shape.width, spec.shape.height, spec.filter);
            }
        }

        if let Some(color) = &self.recolor {
            let color: ColorType = *color;
            image = color_util::convert_to_colortype(image, color);
        }

        txn.set_scalar_output("image", Arc::new(image))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::core::experimental_run_batch_env;
    use crate::core::operations::environment::{OpEnvironment, new_default_operator_environment};
    use crate::core::operations::planner::OperationPlanner;
    use crate::core::rows::RowBatch;
    use crate::core::schema::{ColumnSchema, FirehoseTableSchema};
    use crate::ops::image::test_util::{assert_image_close, generate_gradient_pattern};
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

        let source_image: DynamicImage = generate_gradient_pattern(ImageShape {
            width: 32,
            height: 32,
        })
        .into();

        source_image
            .save(&image_path)
            .expect("Failed to save test image");

        let env = new_default_operator_environment();

        let mut schema = FirehoseTableSchema::from_columns(&[ColumnSchema::new::<String>("path")]);

        env.apply_plan_to_schema(
            &mut schema,
            OperationPlanner::for_operation_id(LOAD_IMAGE)
                .with_input("path", "path")
                .with_output("image", "image")
                .with_config(ImageLoader::default()),
        )?;

        env.apply_plan_to_schema(
            &mut schema,
            OperationPlanner::for_operation_id(LOAD_IMAGE)
                .with_input("path", "path")
                .with_output("image", "resized_gray")
                .with_config(
                    ImageLoader::default()
                        .with_resize(
                            ResizeSpec::new(ImageShape {
                                width: 16,
                                height: 16,
                            })
                            .with_filter(FilterType::Nearest),
                        )
                        .with_recolor(ColorType::L16),
                ),
        )?;

        let schema = Arc::new(schema);

        let mut batch = RowBatch::with_size(schema.clone(), 1);
        batch[0].set_column(&schema, "path", Some(Arc::new(image_path)));

        experimental_run_batch_env(&mut batch, &env)?;

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
                .resize_exact(16, 16, FilterType::Nearest)
                .to_luma8()
                .into(),
            None,
        );

        Ok(())
    }
}
