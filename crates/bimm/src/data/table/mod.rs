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
        let schema = BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<String>("path"),
            BimmColumnSchema::new::<Vec<u8>>("raw_image")
                .with_build_info("load_image", &["path"], json!(null))
                .with_ephemeral(),
            BimmColumnSchema::new::<Vec<u8>>("image").with_build_info(
                "image_aug",
                &["raw_image"],
                json!({"blur": 0.1, "brightness": 0.2}),
            ),
        ]);

        assert_eq!(
            serde_json::to_string_pretty(&schema).unwrap(),
            indoc! {r#"
                {
                  "columns": [
                    {
                      "name": "path",
                      "data_type": {
                        "type_name": "alloc::string::String"
                      }
                    },
                    {
                      "name": "raw_image",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      },
                      "ephemeral": true,
                      "build_info": {
                        "op_name": "load_image",
                        "deps": [
                          "path"
                        ]
                      }
                    },
                    {
                      "name": "image",
                      "data_type": {
                        "type_name": "alloc::vec::Vec<u8>"
                      },
                      "build_info": {
                        "op_name": "image_aug",
                        "deps": [
                          "raw_image"
                        ],
                        "args": {
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
