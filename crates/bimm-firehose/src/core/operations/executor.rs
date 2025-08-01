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

pub struct ThreadedBatchExecutor {
    workers: usize,
    pool: threadpool::ThreadPool,

    schema: Arc<FirehoseTableSchema>,
    environment: Arc<dyn OpEnvironment>,
    op_runners: Vec<Arc<OperationRunner>>,

    tx: std::sync::mpsc::Sender<(usize, FirehoseRowBatch)>,
    rx: std::sync::mpsc::Receiver<(usize, FirehoseRowBatch)>,
}

impl ThreadedBatchExecutor {
    pub fn new(
        workers: usize,
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

        let pool = threadpool::ThreadPool::new(workers);

        let (tx, rx) = std::sync::mpsc::channel();

        Ok(ThreadedBatchExecutor {
            schema,
            environment,
            workers,
            pool,
            op_runners,
            tx,
            rx,
        })
    }
}

impl FirehoseBatchExecutor for ThreadedBatchExecutor {
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
        let chunk_size = batch.len() / self.workers;
        for idx in 0..self.workers {
            let mut chunk = FirehoseRowBatch::new(batch.schema().clone());
            let k: usize = std::cmp::min(chunk_size, batch.len());
            batch.drain(0..k).for_each(|r| chunk.add_row(r));

            let tx = self.tx.clone();
            let op_runners = self.op_runners.clone();
            self.pool.execute(move || {
                let mut chunk = chunk;
                for runner in &op_runners {
                    runner
                        .apply_to_batch(&mut chunk)
                        .expect("Failed to apply operation");
                }
                tx.send((idx, chunk))
                    .expect("Failed to send processed chunk");
            });
        }

        let mut chunks = Vec::with_capacity(self.workers);
        for _ in 0..self.workers {
            let recieved = self.rx.recv().expect("Failed to receive chunk");
            chunks.push(recieved);
        }
        chunks.sort_by_key(|(idx, _)| *idx);

        chunks
            .into_iter()
            .for_each(|(_, chunk)| batch.append_batch(chunk));

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[allow(dead_code)]
    const SE_IS_SEND: fn() = || {
        fn assert_send<T: Send>() {}
        assert_send::<Vec<Arc<OperationRunner>>>();
    };
}
