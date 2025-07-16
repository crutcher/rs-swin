use crate::data::table::{AnyArc, BimmDataTypeDescription, BimmRow, BimmTableSchema};
use std::any::Any;
use std::collections::BTreeMap;

pub trait BuilderFactory {
    fn create_builder(
        &self,
        dep_types: &BTreeMap<String, BimmDataTypeDescription>,
        params: &BTreeMap<String, serde_json::Value>,
        data_type: &BimmDataTypeDescription,
    ) -> Result<Box<dyn BimmColumnBuilder>, String>;
}

/// BimmColumnBuilder lifecycle
///
pub trait BimmColumnBuilder {
    /// Builds a single row cell.
    ///
    /// ## Arguments
    ///
    /// * `deps`: A map of dependencies where the key is the name of the dependency and the value is an optional `AnyArc` value.
    ///
    /// ## Returns
    ///
    /// An `Option<AnyArc>` representing the built cell value, or an error message if the build fails.
    fn build_cell(
        &self,
        deps: &BTreeMap<String, Option<&dyn Any>>,
    ) -> Result<Option<AnyArc>, String>;
}

pub struct BimmColumnBuilderBinding {
    pub table_schema: BimmTableSchema,
    pub builder: Box<dyn BimmColumnBuilder>,

    pub column_name: String,
    column_index: usize,
    dep_map: Vec<(String, usize)>,
}

impl BimmColumnBuilderBinding {
    pub fn new(
        column_name: &str,
        table_schema: BimmTableSchema,
        builder: Box<dyn BimmColumnBuilder>,
    ) -> Result<Self, String> {
        let column_index = table_schema.check_column_index(column_name).unwrap();
        let column_schema = &table_schema[column_index];

        let build_info = column_schema
            .build_info
            .as_ref()
            .ok_or_else(|| format!("Column '{column_name}' does not have build info"))?;

        let dep_map: Vec<(String, usize)> = build_info
            .deps
            .iter()
            .map(|(pname, cname)| {
                (
                    pname.clone(),
                    table_schema.check_column_index(cname).unwrap(),
                )
            })
            .collect();

        Ok(Self {
            table_schema,
            builder,
            column_name: column_name.to_string(),
            column_index,
            dep_map,
        })
    }

    pub fn build_rows(
        &self,
        rows: &mut [BimmRow],
    ) -> Result<(), String> {
        // This is the builder call.
        for row in rows.iter_mut() {
            // Collect the dep column values.
            // TODO: extract method
            let deps = {
                let mut deps = BTreeMap::new();
                for (pname, index) in &self.dep_map {
                    deps.insert(pname.clone(), row.get_untyped_slot(*index));
                }
                deps
            };

            row.set_slot(self.column_index, self.builder.build_cell(&deps)?);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {}
