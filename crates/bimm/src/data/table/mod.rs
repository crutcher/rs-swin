/// Identifier pattern checkers.
mod identifiers;

/// Bimm Table Rows.
mod rows;

// Bimm Table Batch.
mod batch;
/// Bimm Table Schema.
mod schema;

pub use batch::*;
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
        let mut schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<String>("path").with_description("path to the image")
        ]);

        // TODO: building up fluent schema builder pattern; some kind of op/builder to construct this.
        // - bound operator environment (name <-> implementation mapping)
        // - symbolic column def (wth some basic source type / op argument checking ...)
        // - extends column schema with build info

        schema.add_column(
            BimmColumnSchema::new::<String>("class_name")
                .with_description("category class name")
                .with_build_info("path_to_class", &[("path", "source")], json!(null)),
        );
        schema.add_column(
            BimmColumnSchema::new::<u32>("class")
                .with_description("category class code")
                .with_build_info("class_code", &[("class_name", "source")], json!(null)),
        );

        schema.add_column(
            BimmColumnSchema::new::<Vec<u8>>("raw_image")
                .with_description("initial image loaded from disk")
                .with_build_info("load_image", &[("path", "source")], json!(null))
                .with_ephemeral(),
        );

        // same.
        schema.add_column(
            BimmColumnSchema::new::<Vec<u8>>("aug_image")
                .with_description("augmented image")
                .with_build_info(
                    "image_aug",
                    &[("raw_image", "source")],
                    json!({"blur": 0.1, "brightness": 0.2}),
                ),
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
                      "build_info": {
                        "op_name": "path_to_class",
                        "deps": [
                          [
                            "path",
                            "source"
                          ]
                        ]
                      }
                    },
                    {
                      "name": "class",
                      "description": "category class code",
                      "data_type": {
                        "type_name": "u32"
                      },
                      "build_info": {
                        "op_name": "class_code",
                        "deps": [
                          [
                            "class_name",
                            "source"
                          ]
                        ]
                      }
                    },
                    {
                      "name": "raw_image",
                      "description": "initial image loaded from disk",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      },
                      "ephemeral": true,
                      "build_info": {
                        "op_name": "load_image",
                        "deps": [
                          [
                            "path",
                            "source"
                          ]
                        ]
                      }
                    },
                    {
                      "name": "aug_image",
                      "description": "augmented image",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      },
                      "build_info": {
                        "op_name": "image_aug",
                        "deps": [
                          [
                            "raw_image",
                            "source"
                          ]
                        ],
                        "params": {
                          "blur": 0.1,
                          "brightness": 0.2
                        }
                      }
                    }
                  ]
                }"#
            }
        );
    }
}
