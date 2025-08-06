/// Defines legal identifiers for firehose tables.
pub mod identifiers;
/// Defines the operator environment for firehose tables.
pub mod operations;
/// Defines rows and row batches for firehose tables.
pub mod rows;
/// Defines the symbolic schema for firehose tables.
pub mod schema;

/// Defines `ValueBox`, a sum type for Json Values and boxed values.
pub mod valuebox;

// TODO: Work out what the `$crate::core::*` re-exports should be.
pub use rows::{FirehoseRow, FirehoseRowBatch, FirehoseRowReader, FirehoseRowWriter};
pub use schema::FirehoseTableSchema;
pub use valuebox::ValueBox;

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ops::image::loader::{ImageLoader, ResizeSpec};
    use crate::ops::image::tensor_loader::{
        ImageDimLayout, ImgToTensorConfig, TargetDType, image_to_f32_tensor,
    };
    use crate::ops::image::test_util::assert_image_close;
    use crate::ops::image::{ImageShape, test_util};
    use burn::backend::NdArray;
    use burn::prelude::{Shape, Tensor};

    use crate::core::ValueBox;
    use crate::core::operations::executor::{FirehoseBatchExecutor, SequentialBatchExecutor};
    use crate::core::rows::{FirehoseRowReader, FirehoseRowWriter};
    use crate::core::schema::ColumnSchema;
    use crate::ops::init_burn_device_operator_environment;
    use image::imageops::FilterType;
    use image::{ColorType, DynamicImage};
    use indoc::indoc;
    use std::sync::Arc;

    #[derive(serde::Serialize, serde::Deserialize, Clone, Debug)]
    pub struct TensorDescription {
        pub shape: Vec<usize>,
        pub dtype: burn::tensor::DType,
    }

    #[test]
    fn test_example() -> anyhow::Result<()> {
        let temp_dir = tempfile::tempdir().unwrap();

        type B = NdArray;

        let device = Default::default();

        let env = Arc::new(init_burn_device_operator_environment::<B>(&device));

        let schema = {
            let mut schema =
                FirehoseTableSchema::from_columns(&[
                    ColumnSchema::new::<String>("path").with_description("path to the image")
                ]);

            ImageLoader::default()
                .with_resize(
                    ResizeSpec::new(ImageShape {
                        width: 16,
                        height: 24,
                    })
                    .with_filter(FilterType::Nearest),
                )
                .with_recolor(ColorType::L16)
                .to_plan("path", "image")
                .apply_to_schema(&mut schema, env.as_ref())?;

            ImgToTensorConfig::new()
                .with_dtype(TargetDType::F32)
                .with_dim_layout(ImageDimLayout::HWC)
                .to_plan("image", "tensor")
                .apply_to_schema(&mut schema, env.as_ref())?;

            Arc::new(schema)
        };

        let executor = SequentialBatchExecutor::new(schema.clone(), env.clone())?;

        assert_eq!(
            serde_json::to_string_pretty(schema.as_ref()).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "path",
                      "description": "path to the image",
                      "data_type": {
                        "type_name": "alloc::string::String"
                      }
                    },
                    {
                      "name": "image",
                      "description": "Image loaded from disk.",
                      "data_type": {
                        "type_name": "image::dynimage::DynamicImage"
                      }
                    },
                    {
                      "name": "tensor",
                      "description": "Tensor representation of the image.",
                      "data_type": {
                        "type_name": "burn_tensor::tensor::api::base::Tensor<burn_ndarray::backend::NdArray, 3>"
                      }
                    }
                  ],
                  "build_plans": [
                    {
                      "operator_id": "bimm_firehose::ops::image::loader::LOAD_IMAGE",
                      "description": "Loads an image from disk.",
                      "config": {
                        "recolor": "L16",
                        "resize": {
                          "filter": "Nearest",
                          "shape": {
                            "height": 24,
                            "width": 16
                          }
                        }
                      },
                      "inputs": {
                        "path": "path"
                      },
                      "outputs": {
                        "image": "image"
                      }
                    },
                    {
                      "operator_id": "bimm_firehose::ops::image::tensor_loader::IMAGE_TO_TENSOR",
                      "description": "Converts an image to a tensor.",
                      "config": {
                        "dim_layout": "HWC",
                        "dtype": "F32"
                      },
                      "inputs": {
                        "image": "image"
                      },
                      "outputs": {
                        "tensor": "tensor"
                      }
                    }
                  ]
                }"#,
            }
        );

        let mut batch = FirehoseRowBatch::new_with_size(schema.clone(), 1);

        let source_image: DynamicImage = test_util::generate_gradient_pattern(ImageShape {
            width: 32,
            height: 32,
        })
        .into();

        {
            let image_path = temp_dir
                .path()
                .join("test.png")
                .to_string_lossy()
                .to_string();

            source_image
                .save(&image_path)
                .expect("Failed to save test image");

            batch[0].set("path", ValueBox::serializing(image_path)?);
        }

        executor.execute_batch(&mut batch)?;

        let row = &batch[0];

        let row_image = row.get("image").unwrap().as_ref::<DynamicImage>()?;
        assert_image_close(
            row_image,
            &source_image
                .resize_exact(16, 24, FilterType::Nearest)
                .to_luma8()
                .into(),
            None,
        );

        let row_tensor = row.get("tensor").unwrap().as_ref::<Tensor<B, 3>>()?;
        assert_eq!(row_tensor.dtype(), burn::tensor::DType::F32);
        assert_eq!(row_tensor.shape(), Shape::new([24, 16, 1]));
        row_tensor.to_data().assert_eq(
            &image_to_f32_tensor::<B>(row_image, &device).to_data(),
            true,
        );

        Ok(())
    }
}
