use crate::core::operations::environment::OpEnvironment;
use crate::core::operations::operator::OperationRunner;
use crate::core::rows::RowBatch;
use std::sync::Arc;

/// Trait for executing a batch of operations on a `RowBatch`.
pub trait FirehoseBatchExecutor {
    /// Runs the butch under the policy of the executor.
    fn execute_batch(
        &self,
        batch: &mut RowBatch,
    ) -> anyhow::Result<()>;
}

/// A sequential batch executor.
///
/// Runs every `BuildPlan` in the batch schema;
/// executes serially with no threading.
pub struct SequentialBatchExecutor {
    /// The operator environment to use for executing the batch.
    environment: Arc<dyn OpEnvironment + 'static>,
}

impl SequentialBatchExecutor {
    /// Creates a new `DefaultBatchExecutor` with the given operator environment.
    pub fn new(environment: Arc<dyn OpEnvironment>) -> Self {
        SequentialBatchExecutor { environment }
    }
}

impl FirehoseBatchExecutor for SequentialBatchExecutor {
    fn execute_batch(
        &self,
        batch: &mut RowBatch,
    ) -> anyhow::Result<()> {
        let schema = batch.schema.clone();

        let (_base, plans) = schema.build_order()?;
        // TODO: ensure that the base is present in the batch rows.

        for plan in &plans {
            let plan = Arc::new(plan.clone());
            let runner =
                OperationRunner::new_for_plan(schema.clone(), plan, self.environment.as_ref())?;
            runner.apply_to_batch(batch)?;
        }
        Ok(())
    }
}
