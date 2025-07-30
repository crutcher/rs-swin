use crate::core::operations::signature::{FirehoseOperatorSignature, ParameterArity};
use crate::core::rows::{AnyArc, Row, RowBatch};
use crate::core::schema::{BuildPlan, FirehoseTableSchema};
use anyhow::bail;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::sync::Arc;

/// Scheduling metadata for an operator.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct OperatorSchedulingMetadata {
    /// The largest effective batch size for the operator.
    pub effective_batch_size: usize,
}

/// An instantiated column => column operator.
pub trait FirehoseOperator: 'static + Send + Sync + Debug {
    /// Gets the scheduling metadata for the operator.
    fn scheduling_metadata(&self) -> OperatorSchedulingMetadata {
        OperatorSchedulingMetadata {
            // By default, we process one row at a time.
            effective_batch_size: 1,
        }
    }

    /// Apply the operator to a batch of rows in the provided context.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the operation fails.
    #[must_use]
    fn apply_to_batch(
        &self,
        txn: &mut FirehoseBatchTransaction,
    ) -> anyhow::Result<()> {
        for index in 0..txn.len() {
            let mut row_txn = txn.row_transaction(index);
            self.apply_to_row(&mut row_txn)?;
            txn.commit_row_transaction(&row_txn)?;
        }
        Ok(())
    }

    /// Apply the operator to a single row in the provided context.
    ///
    /// Implementations which override `apply_batch` should leave this as unimplemented.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the operation fails.
    #[must_use]
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        let _ignored = txn;
        unimplemented!()
    }
}

/// A transaction for a batch of rows processed by an operator.
#[derive(Debug)]
pub struct FirehoseBatchTransaction {
    /// The transaction's copy of the row batch.
    ///
    /// This is local, and will be commited or discarded atomically.
    pub row_batch: RowBatch,

    /// The build plan that describes the operator and its inputs/outputs.
    pub build_plan: Arc<BuildPlan>,

    /// Signature of the operator being executed.
    pub signature: Arc<FirehoseOperatorSignature>,
}

impl FirehoseBatchTransaction {
    /// Creates a new `OperatorBatchTransaction` for the given row batch.
    pub fn new(
        row_batch: RowBatch,
        build_plan: Arc<BuildPlan>,
        signature: Arc<FirehoseOperatorSignature>,
    ) -> FirehoseBatchTransaction {
        FirehoseBatchTransaction {
            row_batch,
            build_plan,
            signature,
        }
    }

    /// The length of the row batch.
    pub fn len(&self) -> usize {
        self.row_batch.len()
    }

    /// Checks if the row batch is empty.
    pub fn is_empty(&self) -> bool {
        self.row_batch.is_empty()
    }

    /// Constructs a row transaction for a specific index in the row batch.
    pub fn row_transaction(
        &self,
        index: usize,
    ) -> FirehoseRowTransaction {
        FirehoseRowTransaction {
            schema: self.row_batch.schema.clone(),
            build_plan: self.build_plan.clone(),
            row: self.row_batch[index].clone(),
            signature: self.signature.clone(),
            index,
        }
    }

    /// Commits a row back to the row batch at the specified index.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the index is out of bounds.
    pub fn commit_row(
        &mut self,
        row: &Row,
        index: usize,
    ) -> anyhow::Result<()> {
        if index >= self.row_batch.len() {
            bail!(
                "Index {index} out of bounds for row batch of length {}",
                self.row_batch.len()
            );
        }
        self.row_batch[index].assign_from(row);
        Ok(())
    }

    /// Commits a row transaction back to the row batch.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the operation fails.
    pub fn commit_row_transaction(
        &mut self,
        row_txn: &FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        self.commit_row(&row_txn.row, row_txn.index)
    }
}

/// Represents a row-level transaction for an operator.
pub struct FirehoseRowTransaction {
    /// The schema of the table being processed.
    pub schema: Arc<FirehoseTableSchema>,

    /// The build plan that describes the operator and its inputs/outputs.
    pub build_plan: Arc<BuildPlan>,

    /// Signature of the operator being executed.
    pub signature: Arc<FirehoseOperatorSignature>,

    /// The transaction's copy of the row.
    pub row: Row,

    /// The index of the row in the batch.
    pub index: usize,
}

impl FirehoseRowTransaction {
    /// Gets a down-casted required scalar value from the row at the specified index.
    ///
    /// # Generic Parameters
    ///
    /// * `T` - The type to which the value should be down-casted.
    ///
    /// # Arguments
    ///
    /// * `parameter_name` - The name of the parameter to retrieve.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<&T>` containing a reference to the value of type `T` if found.
    #[must_use]
    pub fn get_required_scalar_input<T: 'static>(
        &self,
        parameter_name: &str,
    ) -> anyhow::Result<&T> {
        let column_name = self.build_plan.translate_input_name(parameter_name)?;

        let parameter_sig = self
            .signature
            .get_input_parameter(parameter_name)
            .unwrap_or_else(|| panic!("Required input '{parameter_name}' not found in signature"));
        if parameter_sig.arity != ParameterArity::Required {
            bail!("Input '{parameter_name}' is not required in operator signature",);
        }

        match self
            .row
            .get_column_checked::<T>(&self.schema, column_name)?
        {
            Some(value) => Ok(value),
            None => bail!(
                "Required input '{}' not found in row at index {}",
                parameter_name,
                self.index
            ),
        }
    }

    /// Sets a scalar output.
    ///
    /// # Arguments
    ///
    /// * `parameter_name` - The name of the output parameter to set.
    /// * `value` - The value to set for the output parameter, wrapped in `AnyArc`.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<()>` indicating success or containing an error if the operation fails.
    #[must_use]
    pub fn set_scalar_output(
        &mut self,
        parameter_name: &str,
        value: AnyArc,
    ) -> anyhow::Result<()> {
        let column_name = self.build_plan.translate_output_name(parameter_name)?;

        match self.signature.get_output_parameter(parameter_name) {
            Some(_) => (),
            None => bail!("Output parameter '{parameter_name}' not found in signature"),
        }

        self.row.set_column(&self.schema, column_name, Some(value));

        Ok(())
    }
}
