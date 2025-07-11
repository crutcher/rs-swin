use crate::data::error::DataLoadError;
use std::fmt::Debug;

pub trait DataLoader<M, V>
where
    M: Debug + Clone + Send + Sync,
    V: Debug + Clone + Send + Sync,
{
    fn load(
        &self,
        meta: &M,
    ) -> Result<V, DataLoadError>;
}

impl<M, V, F> DataLoader<M, V> for F
where
    M: Debug + Clone + Send + Sync,
    V: Debug + Clone + Send + Sync,
    F: Fn(&M) -> Result<V, DataLoadError> + Send + Sync,
{
    fn load(
        &self,
        meta: &M,
    ) -> Result<V, DataLoadError> {
        self(meta)
    }
}
