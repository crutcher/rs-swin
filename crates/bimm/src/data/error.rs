use std::fmt::{Debug, Display};

#[derive(Debug, PartialEq, Eq)]
pub enum DataLoadError {
    NotFound { id: String, meta: String },
}

impl Display for DataLoadError {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            DataLoadError::NotFound { id, meta } => {
                write!(f, "Data not found: id = {id}, meta = {meta}")?;
                Ok(())
            }
        }
    }
}

impl std::error::Error for DataLoadError {}

impl DataLoadError {
    pub fn not_found<M>(
        id: &str,
        meta: &M,
    ) -> Self
    where
        M: Debug,
    {
        DataLoadError::NotFound {
            id: id.to_string(),
            meta: format!("{meta:?}"),
        }
    }
}
