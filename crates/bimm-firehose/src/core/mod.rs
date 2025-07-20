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
    use crate::core::op_spec::ParameterSpec;
    use crate::ops::image::loader::{ImageLoader, ImageLoaderFactory};
    use indoc::indoc;

    #[test]
    fn test_example() -> Result<(), String> {
        let path_to_class_op_id: OperatorId = ("example", "path_to_class").into();
        let path_to_class_spec: OperatorSpec = OperatorSpec::new()
            .with_description("Extracts class name from image path")
            .with_input(
                ParameterSpec::new::<String>("path").with_description("Path to segment for class."),
            )
            .with_output(
                ParameterSpec::new::<String>("name").with_description("category class name"),
            )
            .with_output(ParameterSpec::new::<u32>("code").with_description("category class code"));

        let mut schema = TableSchema::from_columns(&[
            ColumnSchema::new::<String>("path").with_description("path to the image")
        ]);

        experimental_plan_columns(
            &mut schema,
            &path_to_class_op_id,
            &path_to_class_spec,
            &[("path", "path")],
            &[("name", "class_name"), ("code", "class_code")],
            None as Option<()>,
        )?;

        experimental_plan_columns(
            &mut schema,
            &ImageLoaderFactory::load_image_op_id(),
            &ImageLoaderFactory::load_image_op_spec(),
            &[("path", "path")],
            &[("image", "raw_image")],
            Some(ImageLoader::new()),
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
                      "name": "class_name",
                      "description": "category class name",
                      "data_type": {
                        "type_name": "alloc::string::String"
                      }
                    },
                    {
                      "name": "class_code",
                      "description": "category class code",
                      "data_type": {
                        "type_name": "u32"
                      }
                    },
                    {
                      "name": "raw_image",
                      "description": "Image loaded from disk.",
                      "data_type": {
                        "type_name": "image::dynimage::DynamicImage"
                      }
                    }
                  ],
                  "build_plans": [
                    {
                      "operator": {
                        "namespace": "example",
                        "name": "path_to_class"
                      },
                      "description": "Extracts class name from image path",
                      "inputs": {
                        "path": "path"
                      },
                      "outputs": {
                        "code": "class_code",
                        "name": "class_name"
                      }
                    },
                    {
                      "operator": {
                        "namespace": "image",
                        "name": "load_image"
                      },
                      "description": "Loads an image from disk.",
                      "config": {},
                      "inputs": {
                        "path": "path"
                      },
                      "outputs": {
                        "image": "raw_image"
                      }
                    }
                  ]
                }"#,
            }
        );

        Ok(())
    }
}
