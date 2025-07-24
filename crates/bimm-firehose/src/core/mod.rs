mod batch;
mod identifiers;
mod operators;
mod rows;
mod schema;

pub use batch::*;
pub use identifiers::*;
pub use operators::*;
pub use rows::*;
pub use schema::*;

/// Runs a batch of rows through the operator environment, applying the build plans defined in the schema.
pub fn experimental_run_batch_env<E>(
    batch: &mut RowBatch,
    env: &E,
) -> Result<(), String>
where
    E: OpEnvironment,
{
    let schema = batch.schema.as_ref();

    let (_base, plans) = schema.build_order()?;
    // TODO: ensure that the base is present in the batch rows.

    for plan in &plans {
        let builder = ColumnBuilder::new_for_plan(schema, plan, env)?;
        builder.apply_batch(batch.rows.as_mut_slice()).unwrap();
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    use crate::ops::image::loader::{ImageLoader, LOAD_IMAGE, ResizeSpec};
    use crate::ops::image::tensor_loader::{
        IMAGE_TO_TENSOR, ImgToTensorConfig, TargetDType, img_to_tensor_op_binding,
    };
    use crate::ops::image::test_util::assert_image_close;
    use crate::ops::image::{ImageShape, test_util};
    use burn::backend::NdArray;
    use burn::prelude::{Shape, Tensor};

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
    fn test_example() -> Result<(), String> {
        type B = NdArray;

        let device = Default::default();

        let mut env = new_default_operator_environment();
        env.add_binding(img_to_tensor_op_binding::<B>(&device))?;

        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<String>("path").with_description("path to the image")
        ]);

        env.plan_operation(
            &mut schema,
            CallBuilder::new(LOAD_IMAGE)
                .with_input("path", "path")
                .with_output("image", "image")
                .with_output_extension(
                    "image",
                    ImageShape {
                        width: 16,
                        height: 24,
                    },
                )
                .with_config(
                    ImageLoader::default()
                        .with_resize(
                            ResizeSpec::new(ImageShape {
                                width: 16,
                                height: 24,
                            })
                            .with_filter(FilterType::Nearest),
                        )
                        .with_recolor(ColorType::L16),
                ),
        )?;

        env.plan_operation(
            &mut schema,
            CallBuilder::new(IMAGE_TO_TENSOR)
                .with_input("image", "image")
                .with_output("tensor", "tensor")
                .with_output_extension(
                    "tensor",
                    TensorDescription {
                        shape: vec![24, 16, 1],
                        dtype: burn::tensor::DType::F32,
                    },
                )
                .with_config(ImgToTensorConfig::new().with_dtype(TargetDType::F32)),
        )?;

        assert_eq!(
            serde_json::to_string_pretty(&schema).unwrap(),
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
                        "type_name": "image::dynimage::DynamicImage",
                        "extension": {
                          "height": 24,
                          "width": 16
                        }
                      }
                    },
                    {
                      "name": "tensor",
                      "description": "Tensor representation of the image.",
                      "data_type": {
                        "type_name": "burn_tensor::tensor::api::base::Tensor<burn_ndarray::backend::NdArray, 3>",
                        "extension": {
                          "dtype": "F32",
                          "shape": [
                            24,
                            16,
                            1
                          ]
                        }
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

        let mut batch = RowBatch::with_size(schema.clone(), 1);

        let source_image: DynamicImage = test_util::generate_gradient_pattern(ImageShape {
            width: 32,
            height: 32,
        })
        .into();

        let temp_dir = tempfile::tempdir().expect("Failed to create temporary directory");
        {
            let image_path = temp_dir
                .path()
                .join("test.png")
                .to_string_lossy()
                .to_string();

            source_image
                .save(&image_path)
                .expect("Failed to save test image");

            batch[0].set_column(&schema, "path", Some(Arc::new(image_path)));
        }

        experimental_run_batch_env(&mut batch, &env)?;

        let loaded_image: &DynamicImage = batch[0]
            .get_column(&schema, "image")
            .expect("Failed to get loaded image");
        assert_image_close(
            loaded_image,
            &source_image
                .resize_exact(16, 24, FilterType::Nearest)
                .to_luma8()
                .into(),
            None,
        );

        let loaded_tensor: &Tensor<B, 3> = batch[0].get_column_checked(&schema, "tensor")?.unwrap();

        assert_eq!(loaded_tensor.dtype(), burn::tensor::DType::F32);
        assert_eq!(loaded_tensor.shape(), Shape::new([24, 16, 1]));

        Ok(())
    }
}
