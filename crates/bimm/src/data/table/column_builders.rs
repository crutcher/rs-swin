use crate::data::table::{AnyArc, BimmDataTypeDescription, BimmRow, BimmTableSchema};
use std::any::Any;
use std::collections::BTreeMap;
use std::sync::Arc;

/// Factory for creating `BimmColumnFunc` instances.
pub trait BimmColumnFuncFactory {
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
    fn init_func(
        &self,
        op_name: &str,
        dep_types: &BTreeMap<String, BimmDataTypeDescription>,
        params: &BTreeMap<String, serde_json::Value>,
        data_type: &BimmDataTypeDescription,
    ) -> Result<Arc<dyn BimmColumnFunc>, String>;
}

/// Factory for creating `BimmColumnFunc` instances based on operation names.
#[derive(Clone, Default)]
pub struct MapColumnFuncFactory {
    /// A map of operation names to their corresponding `BimmColumnFuncFactory` implementations.
    pub bindings: BTreeMap<String, Arc<dyn BimmColumnFuncFactory>>,
}

impl MapColumnFuncFactory {
    /// Creates a new `MapColumnBuilderFactory` with the given bindings.
    ///
    /// ## Arguments
    ///
    /// * `bindings`: A map of operation names to their corresponding `BimmColumnBuilderFactory` implementations.
    pub fn new(bindings: BTreeMap<String, Arc<dyn BimmColumnFuncFactory>>) -> Self {
        MapColumnFuncFactory { bindings }
    }

    /// Adds a binding for a specific operation name.
    ///
    /// ## Arguments
    ///
    /// * `op_name`: The name of the operation to bind.
    /// * `factory`: The factory that creates the column function for the operation.
    pub fn add_binding(
        &mut self,
        op_name: String,
        factory: Arc<dyn BimmColumnFuncFactory>,
    ) {
        self.bindings.insert(op_name, factory);
    }
}

impl BimmColumnFuncFactory for MapColumnFuncFactory {
    fn init_func(
        &self,
        op_name: &str,
        dep_types: &BTreeMap<String, BimmDataTypeDescription>,
        params: &BTreeMap<String, serde_json::Value>,
        data_type: &BimmDataTypeDescription,
    ) -> Result<Arc<dyn BimmColumnFunc>, String> {
        if let Some(factory) = self.bindings.get(op_name) {
            factory.init_func(op_name, dep_types, params, data_type)
        } else {
            Err(format!("No builder found for operation '{op_name}'"))
        }
    }
}

/// BimmColumnBuilder lifecycle
///
pub trait BimmColumnFunc {
    /// Builds a single row cell.
    ///
    /// ## Arguments
    ///
    /// * `deps`: A map of dependencies where the key is the name of the dependency and the value is an optional `AnyArc` value.
    ///
    /// ## Returns
    ///
    /// An `Option<AnyArc>` representing the built cell value, or an error message if the build fails.
    fn apply(
        &self,
        deps: &BTreeMap<&str, Option<&dyn Any>>,
    ) -> Result<Option<AnyArc>, String>;
}

/// A bound column builder for a specific table schema and column name.
pub struct BimmColumnBuilder {
    /// The table schema to which this column belongs.
    pub table_schema: BimmTableSchema,

    /// The name of the column being built.
    pub column_name: String,

    /// The index of the column in the table schema.
    slot_index: usize,

    /// A map of dependency names to their corresponding slot indices in the table schema.
    slot_map: BTreeMap<String, usize>,

    /// The builder that implements the column function.
    builder: Arc<dyn BimmColumnFunc>,
}

impl BimmColumnBuilder {
    /// Initializes a new `BimmColumnBuilder` for a specific column in the table schema.
    ///
    /// ## Arguments
    ///
    /// * `column_name`: The name of the column to build.
    /// * `table_schema`: The schema of the table to which the column belongs.
    /// * `factory`: A factory that creates the column function for the specified operation.
    ///
    /// ## Returns
    ///
    /// A `Result` containing the initialized `BimmColumnBuilder` if successful, or an error message if initialization fails.
    pub fn init<F>(
        column_name: &str,
        table_schema: BimmTableSchema,
        factory: F,
    ) -> Result<Self, String>
    where
        F: BimmColumnFuncFactory,
    {
        let slot_index = table_schema.check_column_index(column_name)?;
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

        let dep_types: BTreeMap<String, BimmDataTypeDescription> = build_info
            .deps
            .iter()
            .map(|(pname, cname)| {
                let dtype = table_schema[cname.as_ref()].data_type.clone();
                (pname.clone(), dtype)
            })
            .collect();

        let builder = factory
            .init_func(
                &build_info.op,
                &dep_types,
                &build_info.params,
                &column_schema.data_type,
            )
            .map_err(|e| format!("Failed to create builder for column '{column_name}': {e}"))?;

        Ok(BimmColumnBuilder {
            table_schema,
            column_name: column_name.to_string(),
            slot_index,
            slot_map,
            builder,
        })
    }

    /// Applies the builder to a batch of rows.
    ///
    /// ## Arguments
    ///
    /// * `rows`: A mutable slice of `BimmRow` instances that will be modified.
    ///
    /// ## Returns
    ///
    /// A `Result` indicating success or an error message if the batch build fails.
    pub fn build_batch(
        &self,
        rows: &mut [BimmRow],
    ) -> Result<(), String> {
        // TODO: extensions.
        // - smart up-switching to batch-builders.

        for row in rows.iter_mut() {
            self.build_row(row)?;
        }
        Ok(())
    }

    /// Applies the builder to a single row.
    ///
    /// ## Arguments
    ///
    /// * `row`: A mutable reference to the `BimmRow` that will be modified.
    ///
    /// ## Returns
    ///
    /// A `Result` indicating success or an error message if the build fails.
    pub fn build_row(
        &self,
        row: &mut BimmRow,
    ) -> Result<(), String> {
        // TODO: extensions.
        // - policy based enforcement for missing fields.

        let deps = self
            .slot_map
            .iter()
            .map(|(pname, slot)| (pname.as_ref(), row.get_untyped_slot(*slot)))
            .collect::<BTreeMap<_, _>>();

        let value = self.builder.apply(&deps)?;

        row.set_slot(self.slot_index, value);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::data::table::{
        AnyArc, BimmColumnBuilder, BimmColumnFunc, BimmColumnFuncFactory, BimmColumnSchema,
        BimmDataTypeDescription, BimmRowBatch, BimmTableSchema,
    };
    use std::any::Any;
    use std::collections::BTreeMap;
    use std::sync::Arc;

    struct AddFunc {
        bias: i32,
    }

    impl BimmColumnFunc for AddFunc {
        fn apply(
            &self,
            deps: &BTreeMap<&str, Option<&dyn Any>>,
        ) -> Result<Option<AnyArc>, String> {
            let x: i32 = deps
                .values()
                .map(|v| v.unwrap().downcast_ref::<i32>().unwrap())
                .sum();
            let x: i32 = x + self.bias;
            Ok(Some(Arc::new(x) as AnyArc))
        }
    }

    struct AddFuncFactory {}

    impl BimmColumnFuncFactory for AddFuncFactory {
        fn init_func(
            &self,
            _op_name: &str,
            _dep_types: &BTreeMap<String, BimmDataTypeDescription>,
            params: &BTreeMap<String, serde_json::Value>,
            _data_type: &BimmDataTypeDescription,
        ) -> Result<Arc<dyn BimmColumnFunc>, String> {
            let bias = params.get("bias").unwrap().as_i64().unwrap() as i32;
            Ok(Arc::new(AddFunc { bias }))
        }
    }

    #[test]
    fn test_add_func() {
        let schema = Arc::new(BimmTableSchema::from_columns(&[
            BimmColumnSchema::new::<i32>("a"),
            BimmColumnSchema::new::<i32>("b"),
            BimmColumnSchema::new::<i32>("c").with_build_info(
                "add",
                &[("x", "a"), ("y", "b")],
                &[("bias", serde_json::json!(10))],
            ),
        ]));

        let factory = AddFuncFactory {};

        let builder = BimmColumnBuilder::init("c", schema.as_ref().clone(), factory)
            .expect("Failed to initialize column builder");

        let mut batch = BimmRowBatch::with_size(schema.clone(), 2);
        batch[0].set_columns(schema.as_ref(), &["a", "b"], [Arc::new(1), Arc::new(2)]);
        batch[1].set_columns(schema.as_ref(), &["a", "b"], [Arc::new(5), Arc::new(6)]);

        builder
            .build_batch(&mut batch.rows)
            .expect("Failed to build row");

        assert_eq!(
            batch.collect_column_values::<i32>("c"),
            vec![Some(&13), Some(&21)]
        );
        assert_eq!(batch[0].get_column(schema.as_ref(), "c"), Some(&13));
        assert_eq!(batch[1].get_column(schema.as_ref(), "c"), Some(&21));
    }
}
