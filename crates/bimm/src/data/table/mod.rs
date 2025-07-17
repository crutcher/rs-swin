mod batch;
mod column_builders;
mod identifiers;
mod rows;
mod schema;

pub use batch::*;
pub use column_builders::*;
pub use identifiers::*;
pub use rows::*;
pub use schema::*;

#[cfg(test)]
mod tests {
    use super::*;
    use indoc::indoc;
    use serde_json::json;

    #[test]
    fn test_example() {
        // TODO: improve relationship between the symbolic builder mechanism,
        // used here to define build info; and the column builder machinery,
        // used to actually build the columns.

        fn plan_class_extraction(
            schema: &mut BimmTableSchema,
            path_column: &str,
            class_name_column: &str,
            class_code_column: &str,
        ) {
            schema.add_column(
                BimmColumnSchema::new::<String>(class_name_column)
                    .with_description("category class name"),
            );
            schema.add_column(
                BimmColumnSchema::new::<u32>(class_code_column)
                    .with_description("category class code"),
            );
            schema
                .add_build_plan(BuildPlan {
                    operator: OperatorSpec {
                        id: ColumnOperatorId {
                            namespace: "example".to_string(),
                            name: "path_to_class".to_string(),
                        },
                        description: Some("Extracts class name from image path".to_string()),
                        config: json!(null),
                    },
                    inputs: [("source", path_column)]
                        .iter()
                        .map(|(p, c)| (p.to_string(), c.to_string()))
                        .collect(),
                    outputs: [("name", class_name_column), ("code", class_code_column)]
                        .iter()
                        .map(|(p, c)| (p.to_string(), c.to_string()))
                        .collect(),
                })
                .unwrap()
        }

        type ImageStandIn = Vec<u8>;
        fn plan_image_load(
            schema: &mut BimmTableSchema,
            path_column: &str,
            image_column: &str,
        ) {
            schema.add_column(
                BimmColumnSchema::new::<ImageStandIn>(image_column)
                    .with_description("Image loaded from disk"),
            );
            schema
                .add_build_plan(BuildPlan {
                    operator: OperatorSpec {
                        id: ColumnOperatorId {
                            namespace: "example".to_string(),
                            name: "load_image".to_string(),
                        },
                        description: Some("Loads image from disk".to_string()),
                        config: json!(null),
                    },
                    inputs: [("path", path_column)]
                        .iter()
                        .map(|(p, c)| (p.to_string(), c.to_string()))
                        .collect(),
                    outputs: [("image", image_column)]
                        .iter()
                        .map(|(p, c)| (p.to_string(), c.to_string()))
                        .collect(),
                })
                .unwrap()
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        struct ImageAugConfig {
            blur: f64,
            brightness: f64,
        }

        fn plan_image_augmentation(
            schema: &mut BimmTableSchema,
            source_column: &str,
            output_column: &str,
            config: ImageAugConfig,
        ) {
            schema.add_column(
                BimmColumnSchema::new::<Vec<u8>>(output_column).with_description("augmented image"),
            );

            schema
                .add_build_plan(BuildPlan {
                    operator: OperatorSpec {
                        id: ColumnOperatorId {
                            namespace: "example".to_string(),
                            name: "image_aug".to_string(),
                        },
                        description: Some("Augments image with blur and brightness".to_string()),
                        config: serde_json::to_value(config).unwrap(),
                    },
                    inputs: [("source", source_column)]
                        .iter()
                        .map(|(p, c)| (p.to_string(), c.to_string()))
                        .collect(),
                    outputs: [("augmented", output_column)]
                        .iter()
                        .map(|(p, c)| (p.to_string(), c.to_string()))
                        .collect(),
                })
                .unwrap();
        }

        let mut schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<String>("path").with_description("path to the image")
        ]);

        // TODO: building up fluent schema builder pattern; some kind of op/builder to construct this.
        // - bound operator environment (name <-> implementation mapping)
        // - symbolic column def (wth some basic source type / op argument checking ...)
        // - extends column schema with build info

        // TODO: garbage collection for intermediate columns.
        // - could mark columns as intermediate (did, but removed for now).
        // - what pins an intermediate column?
        //   - in the dep graph for missing non-intermediate columns?

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
                      "description": "Image loaded from disk",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      }
                    },
                    {
                      "name": "aug_image",
                      "description": "augmented image",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      }
                    }
                  ],
                  "build_plans": [
                    {
                      "operator": {
                        "id": {
                          "namespace": "example",
                          "name": "path_to_class"
                        },
                        "description": "Extracts class name from image path"
                      },
                      "inputs": {
                        "source": "path"
                      },
                      "outputs": {
                        "code": "class_code",
                        "name": "class_name"
                      }
                    },
                    {
                      "operator": {
                        "id": {
                          "namespace": "example",
                          "name": "load_image"
                        },
                        "description": "Loads image from disk"
                      },
                      "inputs": {
                        "path": "path"
                      },
                      "outputs": {
                        "image": "raw_image"
                      }
                    },
                    {
                      "operator": {
                        "id": {
                          "namespace": "example",
                          "name": "image_aug"
                        },
                        "description": "Augments image with blur and brightness",
                        "config": {
                          "blur": 2.0,
                          "brightness": 0.5
                        }
                      },
                      "inputs": {
                        "source": "raw_image"
                      },
                      "outputs": {
                        "augmented": "aug_image"
                      }
                    }
                  ]
                }"#
            }
        );
    }
}
