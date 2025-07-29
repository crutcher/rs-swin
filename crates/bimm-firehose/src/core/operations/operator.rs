use crate::core::operations::runner::{OperatorApplyBatchContext, OperatorApplyRowContext};
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
