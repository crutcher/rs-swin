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

    #[test]
    fn test_example() {
        // TODO: improve relationship between the symbolic builder mechanism,
        // used here to define build info; and the column builder machinery,
        // used to actually build the columns.

        fn extract_class_name_from_path(
            schema: &BimmTableSchema,
            path_column: &str,
        ) -> ColumnBuildSpec {
            // TODO: better lookup / type check error handling as utils on schema.
            let data_type = &schema[path_column].data_type;
            if data_type.type_name != std::any::type_name::<String>() {
                panic!(
                    "Expected column '{}' to be of type String, found '{}'",
                    path_column, data_type.type_name
                );
            }

            ColumnBuildSpec::new("path_to_class", &[("source", path_column)])
        }
        fn class_name_to_code(
            schema: &BimmTableSchema,
            class_name_column: &str,
        ) -> ColumnBuildSpec {
            let data_type = &schema[class_name_column].data_type;
            if data_type.type_name != std::any::type_name::<String>() {
                panic!(
                    "Expected column '{}' to be of type String, found '{}'",
                    class_name_column, data_type.type_name
                );
            }

            ColumnBuildSpec::new("class_code", &[("source", class_name_column)])
        }

        fn load_image(
            schema: &BimmTableSchema,
            path_column: &str,
        ) -> ColumnBuildSpec {
            let data_type = &schema[path_column].data_type;
            if data_type.type_name != std::any::type_name::<String>() {
                panic!(
                    "Expected column '{}' to be of type String, found '{}'",
                    path_column, data_type.type_name
                );
            }

            ColumnBuildSpec::new("load_image", &[("source", path_column)])
        }

        #[derive(serde::Serialize, serde::Deserialize)]
        struct ImageAugConfig {
            blur: f64,
            brightness: f64,
        }

        fn augment_image(
            schema: &BimmTableSchema,
            raw_image_column: &str,
            config: ImageAugConfig,
        ) -> ColumnBuildSpec {
            let data_type = &schema[raw_image_column].data_type;
            if data_type.type_name != std::any::type_name::<Vec<u8>>() {
                panic!(
                    "Expected column '{}' to be of type Vec<u8>, found '{}'",
                    raw_image_column, data_type.type_name
                );
            }

            ColumnBuildSpec::new("image_aug", &[("source", raw_image_column)]).with_config(config)
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

        schema.add_column(
            BimmColumnSchema::new::<String>("class_name")
                .with_description("category class name")
                .with_build_spec(extract_class_name_from_path(&schema, "path")),
        );
        schema.add_column(
            BimmColumnSchema::new::<u32>("class")
                .with_description("category class code")
                .with_build_spec(class_name_to_code(&schema, "class_name")),
        );

        schema.add_column(
            BimmColumnSchema::new::<Vec<u8>>("raw_image")
                .with_description("initial image loaded from disk")
                .with_build_spec(load_image(&schema, "path")),
        );

        // same.
        schema.add_column(
            BimmColumnSchema::new::<Vec<u8>>("aug_image")
                .with_description("augmented image")
                .with_build_spec(augment_image(
                    &schema,
                    "raw_image",
                    ImageAugConfig {
                        blur: 2.0,
                        brightness: 0.5,
                    },
                )),
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
                      },
                      "build_spec": {
                        "op": "path_to_class",
                        "deps": {
                          "source": "path"
                        }
                      }
                    },
                    {
                      "name": "class",
                      "description": "category class code",
                      "data_type": {
                        "type_name": "u32"
                      },
                      "build_spec": {
                        "op": "class_code",
                        "deps": {
                          "source": "class_name"
                        }
                      }
                    },
                    {
                      "name": "raw_image",
                      "description": "initial image loaded from disk",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      },
                      "build_spec": {
                        "op": "load_image",
                        "deps": {
                          "source": "path"
                        }
                      }
                    },
                    {
                      "name": "aug_image",
                      "description": "augmented image",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      },
                      "build_spec": {
                        "op": "image_aug",
                        "deps": {
                          "source": "raw_image"
                        },
                        "config": {
                          "blur": 2.0,
                          "brightness": 0.5
                        }
                      }
                    }
                  ]
                }"#
            }
        );
    }
}
