use crate::data::table::{AnyArc, BimmDataTypeDescription, BimmRow, BimmTableSchema};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

pub trait BimmColumnBuilderFactory {
    /// Creates a column builder for a specific operation.
    ///
    /// ## Arguments
    ///
    /// * `op_name`: The name of the operation for which to create a builder.
    /// * `dep_types`: A map of dependency types where the key is the name of the dependency and the value is a `BimmDataTypeDescription`.
    /// * `params`: A map of parameters for the operation, where the key is the parameter name and the value is a JSON value.
    /// * `data_type`: The data type description for the column being built.
    ///
    /// ## Returns
    ///
    /// A `Result` containing a boxed `BimmColumnBuilder` if successful, or an error message if the creation fails.
    fn create_builder(
        &self,
        op_name: &str,
        dep_types: &BTreeMap<String, BimmDataTypeDescription>,
        params: &BTreeMap<String, serde_json::Value>,
        data_type: &BimmDataTypeDescription,
    ) -> Result<Box<dyn BimmColumnBuilder>, String>;
}

/// Factory for creating `BimmColumnBuilder` instances based on operation names.
#[derive(Clone, Default)]
pub struct MapColumnBuilderFactory {
    pub bindings: BTreeMap<String, Arc<dyn BimmColumnBuilderFactory>>,
}

impl MapColumnBuilderFactory {
    /// Creates a new `MapColumnBuilderFactory` with the given bindings.
    ///
    /// ## Arguments
    ///
    /// * `bindings`: A map of operation names to their corresponding `BimmColumnBuilderFactory` implementations.
    pub fn new(bindings: BTreeMap<String, Arc<dyn BimmColumnBuilderFactory>>) -> Self {
        MapColumnBuilderFactory { bindings }
    }

    /// Adds a binding for a specific operation name.
    pub fn add_binding(
        &mut self,
        op_name: String,
        factory: Arc<dyn BimmColumnBuilderFactory>,
    ) {
        self.bindings.insert(op_name, factory);
    }
}

impl BimmColumnBuilderFactory for MapColumnBuilderFactory {
    fn create_builder(
        &self,
        op_name: &str,
        dep_types: &BTreeMap<String, BimmDataTypeDescription>,
        params: &BTreeMap<String, serde_json::Value>,
        data_type: &BimmDataTypeDescription,
    ) -> Result<Box<dyn BimmColumnBuilder>, String> {
        if let Some(factory) = self.bindings.get(op_name) {
            factory.create_builder(op_name, dep_types, params, data_type)
        } else {
            Err(format!("No builder found for operation '{op_name}'"))
        }
    }
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
        deps: &BTreeMap<&str, Option<&dyn Any>>,
    ) -> Result<Option<AnyArc>, String>;
}

pub struct BimmColumnBuilderBinding {
    pub table_schema: BimmTableSchema,
    pub builder: Box<dyn BimmColumnBuilder>,

    pub column_name: String,
    slot_index: usize,
    slot_map: BTreeMap<String, usize>,
}

impl BimmColumnBuilderBinding {
    pub fn new(
        column_name: &str,
        table_schema: BimmTableSchema,
        builder: Box<dyn BimmColumnBuilder>,
    ) -> Result<Self, String> {
        let slot_index = table_schema.check_column_index(column_name).unwrap();
        let column_schema = &table_schema[slot_index];

        let build_info = column_schema
            .build_info
            .as_ref()
            .ok_or_else(|| format!("Column '{column_name}' does not have build info"))?;

        let slot_map: BTreeMap<String, usize> = build_info
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
            slot_index,
            slot_map,
        })
    }

    pub fn build_rows(
        &self,
        rows: &mut [BimmRow],
    ) -> Result<(), String> {
        // This is the builder call.
        for row in rows.iter_mut() {
            let deps = self
                .slot_map
                .iter()
                .map(|(pname, slot)| (pname.as_ref(), row.get_untyped_slot(*slot)))
                .collect::<BTreeMap<_, _>>();

            let value = self.builder.build_cell(&deps)?;

            row.set_slot(self.slot_index, value);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {}
