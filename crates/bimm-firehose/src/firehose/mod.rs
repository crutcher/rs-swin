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

#[cfg(test)]
mod tests {

    use super::*;
    use indoc::formatdoc;

    #[test]
    fn test_example() {
        fn plan_class_extraction(
            schema: &mut TableSchema,
            path_column: &str,
            class_name_column: &str,
            class_code_column: &str,
        ) {
            schema
                .add_build_plan_and_outputs(
                    BuildPlan::for_operator(("example", "path_to_class"))
                        .with_description("Extracts class name from image path")
                        .with_inputs(&[("source", path_column)])
                        .with_outputs(&[("name", class_name_column), ("code", class_code_column)]),
                    &[
                        (
                            "name",
                            DataTypeDescription::new::<String>(),
                            "category class name",
                        ),
                        (
                            "code",
                            DataTypeDescription::new::<u32>(),
                            "category class code",
                        ),
                    ],
                )
                .unwrap();
        }

        fn plan_image_load(
            schema: &mut TableSchema,
            path_column: &str,
            image_column: &str,
        ) {
            schema
                .add_build_plan_and_outputs(
                    BuildPlan::for_operator(("example", "load_image"))
                        .with_description("Loads image from disk")
                        .with_inputs(&[("path", path_column)])
                        .with_outputs(&[("image", image_column)]),
                    &[(
                        "image",
                        DataTypeDescription::new::<Vec<u8>>(),
                        "Image loaded from disk",
                    )],
                )
                .unwrap();
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        struct ImageAugConfig {
            blur: f64,
            brightness: f64,
        }

        fn plan_image_augmentation(
            schema: &mut TableSchema,
            source_column: &str,
            output_column: &str,
            config: ImageAugConfig,
        ) {
            schema
                .add_build_plan_and_outputs(
                    BuildPlan::for_operator(("example", "image_aug"))
                        .with_description("Augments image with blur and brightness")
                        .with_config(config)
                        .with_inputs(&[("source", source_column)])
                        .with_outputs(&[("augmented", output_column)]),
                    &[(
                        "augmented",
                        DataTypeDescription::new::<Vec<u8>>(),
                        "augmented image",
                    )],
                )
                .unwrap();
        }

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

        plan_class_extraction(&mut schema, "path", "class_name", "class_code");

        plan_image_load(&mut schema, "path", "raw_image");

        plan_image_augmentation(
            &mut schema,
            "raw_image",
            "aug_image",
            ImageAugConfig {
                blur: 2.0,
                brightness: 0.5,
            },
        );

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
