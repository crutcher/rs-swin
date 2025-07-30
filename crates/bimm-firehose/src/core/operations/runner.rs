use crate::core::operations::environment::BuildPlanContext;
use crate::core::operations::environment::OpEnvironment;
use crate::core::operations::operator::{
    FirehoseBatchTransaction, FirehoseOperator, OperatorSchedulingMetadata,
};
use crate::core::operations::signature::FirehoseOperatorSignature;
use crate::core::rows::RowBatch;
use crate::core::schema::{BuildPlan, FirehoseTableSchema};
use std::fmt::Debug;
use std::sync::Arc;

/// Represents a schema + instantiated column operator for a particular build plan.
pub struct OperationRunner {
    /// The table schema that this operator is bound to.
    pub table_schema: Arc<FirehoseTableSchema>,

    /// A reference to the build plan that this operator is part of.
    pub build_plan: Arc<BuildPlan>,

    /// Signature of the operator.
    pub signature: Arc<FirehoseOperatorSignature>,

    /// The operator that this builder wraps.
    operator: Box<dyn FirehoseOperator>,
}

impl Debug for OperationRunner {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if !f.alternate() {
            f.debug_struct("ColumnBuilder")
                .field("id", &self.build_plan.operator_id)
                .field("inputs", &self.build_plan.inputs)
                .field("outputs", &self.build_plan.outputs)
                .finish()
        } else {
            f.debug_struct("ColumnBuilder")
                .field("build_plan", &self.build_plan)
                .finish()
        }
    }
}

impl OperationRunner {
    /// Create a new `BoundPlanBuilder` by binding a `BuildPlan` to a `BimmTableSchema`.
    ///
    /// # Arguments
    ///
    /// * `table_schema` - The schema of the table to which this plan is bound.
    /// * `build_plan` - The build plan that describes the operator and its inputs/outputs.
    /// * `env` - An environment that can create the operator based on the build plan.
    ///
    /// # Returns
    ///
    /// A result containing a `BoundPlanBuilder` if successful, or an error message if the binding fails.
    #[must_use]
    pub fn new_for_plan<E>(
        table_schema: Arc<FirehoseTableSchema>,
        build_plan: Arc<BuildPlan>,
        env: &E,
    ) -> anyhow::Result<OperationRunner>
    where
        E: OpEnvironment,
    {
        let table_schema = table_schema.clone();
        let build_plan = build_plan.clone();

        let build_plan_context = BuildPlanContext::new(table_schema.clone(), build_plan.clone());

        let signature = Arc::new(
            env.lookup_operator_factory(build_plan_context.operator_id())?
                .signature()
                .clone(),
        );

        let operator = env.init_operator(build_plan_context)?;

        Ok(OperationRunner {
            table_schema,
            build_plan,
            operator,
            signature,
        })
    }

    /// Gets the scheduling metadata for the operator.
    pub fn scheduling_metadata(&self) -> OperatorSchedulingMetadata {
        self.operator.scheduling_metadata()
    }

    /// Apply the operator to a batch of rows.
    ///
    /// # Arguments
    ///
    /// * `rows` - A mutable slice of `BimmRow` instances that will be processed by the operator.
    ///
    /// # Returns
    ///
    /// A result indicating success or an error message if the operation fails.
    pub fn apply_to_batch(
        &self,
        batch: &mut RowBatch,
    ) -> anyhow::Result<()> {
        // TODO: sub-batch based upon scheduling metadata.effective_batch_size.
        // Requires batch.slice(); batch.assign_slice_from();

        let mut txn = FirehoseBatchTransaction::new(
            batch.clone(),
            self.build_plan.clone(),
            self.signature.clone(),
        );

        self.operator.apply_to_batch(&mut txn)?;

        batch.assign_from(&txn.row_batch);

        Ok(())
    }
}
