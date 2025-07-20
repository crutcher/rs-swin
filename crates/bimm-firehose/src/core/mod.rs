mod batch;
mod identifiers;
mod operators;
mod rows;
mod schema;

/// Operator specification utils.
pub mod op_spec;

use crate::core::op_spec::OperatorSpec;
pub use batch::*;
pub use identifiers::*;
pub use operators::*;
pub use rows::*;
pub use schema::*;
use serde::Serialize;
use std::collections::BTreeMap;

/// Driver for executing a batch of build plans against a `RowBatch`.
///
/// This function applies the build plans defined in the `RowBatch` schema to the rows in the batch.
///
/// # Arguments
///
/// * `batch` - A mutable reference to the `RowBatch` containing the rows to be processed.
/// * `factory` - A reference to a factory that can create the operators defined in the build plans.
///
/// # Returns
///
/// A result indicating success or an error message if the operation fails.
pub fn experimental_run_batch<F>(
    batch: &mut RowBatch,
    factory: &F,
) -> Result<(), String>
where
    F: BuildOperatorFactory,
{
    let schema = batch.schema.as_ref();

    let (_base, plans) = schema.build_order()?;
    // TODO: ensure that the base is present in the batch rows.

    for plan in &plans {
        let builder = ColumnBuilder::bind_plan(schema, plan, factory)?;
        builder.apply_batch(batch.rows.as_mut_slice()).unwrap();
    }
    Ok(())
}

/// Plans an operation in the schema, adding a build plan and output columns.
pub fn experimental_plan_columns<I, T>(
    schema: &mut TableSchema,
    operator_id: &I,
    spec: &OperatorSpec,
    input_bindings: &[(&str, &str)],
    output_bindings: &[(&str, &str)],
    config: Option<T>,
) -> Result<(), String>
where
    I: Into<OperatorId> + Clone,
    T: Serialize,
{
    let operator_id: OperatorId = operator_id.to_owned().into();

    let input_types: BTreeMap<String, DataTypeDescription> = input_bindings
        .iter()
        .map(|(pname, cname)| (pname.to_string(), schema[*cname].data_type.clone()))
        .collect();

    spec.validate_inputs(&input_types)?;

    let mut plan = BuildPlan::for_operator(operator_id)
        .with_inputs(input_bindings)
        .with_outputs(output_bindings);

    if let Some(description) = &spec.description {
        plan = plan.with_description(description);
    }
    if let Some(config) = config {
        plan = plan.with_config(config);
    }

    schema.add_build_plan_and_outputs(plan, &spec.output_plan())?;

    Ok(())
}
#[cfg(test)]
mod tests {

    use super::*;
    use indoc::formatdoc;

    #[test]
    fn test_example() {
        // TODO: building up fluent schema builder pattern; some kind of op/builder to construct this.
        // - bound operator environment (name <-> implementation mapping)
        // - symbolic column def (wth some basic source type / op argument checking ...)
        // - extends column schema with build info

        // TODO: garbage collection for intermediate columns.
        // - could mark columns as intermediate (did, but removed for now).
        // - what pins an intermediate column?
        //   - in the dep graph for missing non-intermediate columns?

        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<String>("path").with_description("path to the image")
        ]);

        schema
            .add_build_plan_and_outputs(
                BuildPlan::for_operator(("example", "path_to_class"))
                    .with_description("Extracts class name from image path")
                    .with_inputs(&[("source", "path")])
                    .with_outputs(&[("name", "class_name"), ("code", "class_code")]),
                &[
                    (
                        "name".to_string(),
                        DataTypeDescription::new::<String>(),
                        Some("category class name".to_string()),
                    ),
                    (
                        "code".to_string(),
                        DataTypeDescription::new::<u32>(),
                        Some("category class code".to_string()),
                    ),
                ],
            )
            .expect("failed to add build plan");

        schema
            .add_build_plan_and_outputs(
                BuildPlan::for_operator(("example", "load_image"))
                    .with_description("Loads image from disk")
                    .with_inputs(&[("path", "path")])
                    .with_outputs(&[("image", "raw_image")]),
                &[(
                    "image".to_string(),
                    DataTypeDescription::new::<Vec<u8>>(),
                    Some("Image loaded from disk".to_string()),
                )],
            )
            .expect("failed to add build plan");

        #[derive(serde::Serialize, serde::Deserialize)]
        struct ImageAugConfig {
            blur: f64,
            brightness: f64,
        }
        let config = ImageAugConfig {
            blur: 2.0,
            brightness: 0.5,
        };
        schema
            .add_build_plan_and_outputs(
                BuildPlan::for_operator(("example", "image_aug"))
                    .with_description("Augments image with blur and brightness")
                    .with_config(config)
                    .with_inputs(&[("source", "raw_image")])
                    .with_outputs(&[("augmented", "aug_image")]),
                &[(
                    "augmented".to_string(),
                    DataTypeDescription::new::<Vec<u8>>(),
                    Some("augmented image".to_string()),
                )],
            )
            .expect("failed to add build plan");

        assert_eq!(
            serde_json::to_string_pretty(&schema).unwrap(),
            formatdoc! {r#"
                {{
                  "columns": [
                    {{
                      "name": "path",
                      "description": "path to the image",
                      "data_type": {{
                        "type_name": "alloc::string::String"
                      }}
                    }},
                    {{
                      "name": "class_name",
                      "description": "category class name",
                      "data_type": {{
                        "type_name": "alloc::string::String"
                      }}
                    }},
                    {{
                      "name": "class_code",
                      "description": "category class code",
                      "data_type": {{
                        "type_name": "u32"
                      }}
                    }},
                    {{
                      "name": "raw_image",
                      "description": "Image loaded from disk",
                      "data_type": {{
                        "type_name": "alloc::vec::Vec<u8>"
                      }}
                    }},
                    {{
                      "name": "aug_image",
                      "description": "augmented image",
                      "data_type": {{
                        "type_name": "alloc::vec::Vec<u8>"
                      }}
                    }}
                  ],
                  "build_plans": [
                    {{
                      "id": "{:?}",
                      "operator": {{
                        "namespace": "example",
                        "name": "path_to_class"
                      }},
                      "description": "Extracts class name from image path",
                      "inputs": {{
                        "source": "path"
                      }},
                      "outputs": {{
                        "code": "class_code",
                        "name": "class_name"
                      }}
                    }},
                    {{
                      "id": "{:?}",
                      "operator": {{
                        "namespace": "example",
                        "name": "load_image"
                      }},
                      "description": "Loads image from disk",
                      "inputs": {{
                        "path": "path"
                      }},
                      "outputs": {{
                        "image": "raw_image"
                      }}
                    }},
                    {{
                      "id": "{:?}",
                      "operator": {{
                        "namespace": "example",
                        "name": "image_aug"
                      }},
                      "description": "Augments image with blur and brightness",
                      "config": {{
                        "blur": 2.0,
                        "brightness": 0.5
                      }},
                      "inputs": {{
                        "source": "raw_image"
                      }},
                      "outputs": {{
                        "augmented": "aug_image"
                      }}
                    }}
                  ]
                }}"#,
                schema.build_plans[0].id,
                schema.build_plans[1].id,
                schema.build_plans[2].id,
            }
        );
    }
}
