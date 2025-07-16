use crate::data::table::{
    AnyArc, BimmDataTypeDescription, BimmRow, BimmTableSchema,
};
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
    pub column_name: String,

    pub builder: Box<dyn BimmColumnBuilder>,
}

impl BimmColumnBuilderBinding {
    pub fn new(
        column_name: String,
        builder: Box<dyn BimmColumnBuilder>,
    ) -> Self {
        Self {
            column_name,
            builder,
        }
    }
    fn build_rows(
        &self,
        schema: &BimmTableSchema,
        rows: &mut [BimmRow],
    ) -> Result<(), String> {
        // These are all things which could be cached / verified on init the builder.
        let column_index = schema.check_column_index(&self.column_name)?;
        let column_schema = &schema[column_index];

        let build_info = column_schema.build_info.as_ref().unwrap();
        let dep_map: Vec<(String, usize)> = build_info
            .deps
            .iter()
            .map(|(pname, cname)| (pname.clone(), schema.check_column_index(cname).unwrap()))
            .collect::<Vec<_>>()
            .try_into()
            .map_err(|_| "Invalid build info dependencies")?;

        // This is the builder call.
        for row in rows.iter_mut() {
            // Collect the dep column values.
            // TODO: extract method
            let deps = {
                let mut deps = BTreeMap::new();
                for (pname, index) in &dep_map {
                    deps.insert(pname.clone(), row.get_untyped_slot(*index));
                }
                deps
            };

            row.set_slot(column_index, self.builder.build_cell(&deps)?);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {}
