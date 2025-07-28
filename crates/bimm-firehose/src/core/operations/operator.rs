use crate::core::operations::runner::{OperatorApplyBatchContext, OperatorApplyRowContext};
use crate::core::schema::{BuildPlan, DataTypeDescription, TableSchema};
use std::collections::BTreeMap;
use std::fmt::Debug;

/// An instantiated column => column operator.
pub trait FirehoseOperator: 'static + Send + Sync + Debug {
    /// Get the effective batch size for the operator.
    ///
    /// The default implementation returns 1, indicating that the operator processes one row at a time.
    fn effective_batch_size(&self) -> usize {
        1
    }

    /// Apply the operator to a batch of rows in the provided context.
    #[must_use]
    fn apply_batch(
        &self,
        context: &mut OperatorApplyBatchContext,
    ) -> Result<(), String> {
        // TODO: batch by `effective_batch_size`?
        for idx in 0..context.len() {
            let mut row_context = OperatorApplyRowContext::new(context, idx);
            self.apply_row(&mut row_context)?;
        }
        Ok(())
    }

    /// Apply the operator to a single row in the provided context.
    ///
    /// Implementations which override `apply_batch` should leave this as unimplemented.
    #[must_use]
    fn apply_row(
        &self,
        context: &mut OperatorApplyRowContext,
    ) -> Result<(), String> {
        let _context = context;
        unimplemented!()
    }
}

/// Context for validating and initializing a column build operation.
#[derive(Debug, Clone)]
pub struct OperatorInitializationContext {
    table_schema: TableSchema,

    /// The operator specification for the operator being initialized.
    build_plan: BuildPlan,

    /// A map of input parameter names to their data types.
    input_types: BTreeMap<String, DataTypeDescription>,

    /// A map of output parameter names to their data types.
    output_types: BTreeMap<String, DataTypeDescription>,
}

impl OperatorInitializationContext {
    /// Creates a new `ColumnBuildOperationInitContext` with the specified build plan and input/output types.
    pub fn new(
        table_schema: TableSchema,
        build_plan: BuildPlan,
        input_types: BTreeMap<String, DataTypeDescription>,
        output_types: BTreeMap<String, DataTypeDescription>,
    ) -> Self {
        Self {
            table_schema,
            build_plan,
            input_types,
            output_types,
        }
    }

    /// Returns a reference to the table schema.
    pub fn table_schema(&self) -> &TableSchema {
        &self.table_schema
    }

    /// Returns a reference to the build plan.
    pub fn build_plan(&self) -> &BuildPlan {
        &self.build_plan
    }

    /// Returns a reference to the input types.
    pub fn input_types(&self) -> &BTreeMap<String, DataTypeDescription> {
        &self.input_types
    }

    /// Returns a reference to the output types.
    pub fn output_types(&self) -> &BTreeMap<String, DataTypeDescription> {
        &self.output_types
    }
}
