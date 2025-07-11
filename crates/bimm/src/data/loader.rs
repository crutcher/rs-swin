use crate::data::error::DataLoadError;
use crate::data::schedule::DataLoadMetaDataItem;
use std::fmt::Debug;

pub trait DataLoadDataItem: Debug + Clone + Send + Sync {}

impl<T> DataLoadDataItem for T where T: Debug + Clone + Send + Sync {}

pub trait DataLoadOperator<M, T>: Send + Sync + Debug
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
{
    fn load(
        &self,
        meta: &M,
    ) -> Result<T, DataLoadError>;
}

#[derive(Clone)]
pub struct FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> Result<T, DataLoadError> + Send + Sync,
{
    func: F,

    phantom: std::marker::PhantomData<(M, T)>,
}

impl<M, T, F> Debug for FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> Result<T, DataLoadError> + Send + Sync,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("FnOperator")
            .field("func", &"Function")
            .finish()
    }
}

impl<M, T, F> FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> Result<T, DataLoadError> + Send + Sync,
{
    pub fn new(func: F) -> Self {
        Self {
            func,
            phantom: std::marker::PhantomData,
        }
    }
}

impl<M, T, F> DataLoadOperator<M, T> for FnOperator<M, T, F>
where
    M: DataLoadMetaDataItem,
    T: DataLoadDataItem,
    F: Fn(&M) -> Result<T, DataLoadError> + Send + Sync,
{
    fn load(
        &self,
        meta: &M,
    ) -> Result<T, DataLoadError> {
        (self.func)(meta)
    }
}
