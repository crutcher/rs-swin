use crate::core::operations::environment::OpEnvironment;
use crate::core::operations::operator::OperationRunner;
use crate::core::rows::FirehoseRowBatch;
use crate::core::schema::FirehoseTableSchema;
use std::sync::Arc;

/// Trait for executing a batch of operations on a `RowBatch`.
pub trait FirehoseBatchExecutor {
    /// Returns the schema used by this executor.
    fn schema(&self) -> &Arc<FirehoseTableSchema>;

    /// Returns the operator environment used by this executor.
    fn environment(&self) -> &Arc<dyn OpEnvironment>;

    /// Runs the butch under the policy of the executor.
    fn execute_batch(
        &self,
        batch: &mut FirehoseRowBatch,
    ) -> anyhow::Result<()>;
}

/// A sequential batch executor.
///
/// Runs every `BuildPlan` in the batch schema;
/// executes serially with no threading.
#[derive(Clone)]
pub struct SequentialBatchExecutor {
    /// The schema of the batch to execute.
    schema: Arc<FirehoseTableSchema>,

    /// The operator environment to use for executing the batch.
    environment: Arc<dyn OpEnvironment>,

    /// The operation runners for each plan in the schema.
    op_runners: Vec<Arc<OperationRunner>>,
}

impl SequentialBatchExecutor {
    /// Creates a new `DefaultBatchExecutor` with the given operator environment.
    pub fn new(
        schema: Arc<FirehoseTableSchema>,
        environment: Arc<dyn OpEnvironment>,
    ) -> anyhow::Result<Self> {
        let mut op_runners = Vec::new();
        let (_base, build_order) = schema.build_order()?;
        for plan in &build_order {
            let plan = Arc::new(plan.clone());
            op_runners.push(Arc::new(OperationRunner::new_for_plan(
                schema.clone(),
                plan,
                environment.as_ref(),
            )?));
        }

        Ok(SequentialBatchExecutor {
            schema,
            environment,
            op_runners,
        })
    }
}

impl FirehoseBatchExecutor for SequentialBatchExecutor {
    fn schema(&self) -> &Arc<FirehoseTableSchema> {
        &self.schema
    }

    fn environment(&self) -> &Arc<dyn OpEnvironment> {
        &self.environment
    }

    fn execute_batch(
        &self,
        batch: &mut FirehoseRowBatch,
    ) -> anyhow::Result<()> {
        for runner in &self.op_runners {
            runner.apply_to_batch(batch)?;
        }
        Ok(())
    }
}
