use std::fmt::{Debug, Display};

/// Common error types for data loading operations.
#[derive(Debug, PartialEq, Eq)]
pub enum DataLoadError {
    /// Indicates that the requested data was not found.
    NotFound {
        /// The identifier of the data that was not found.
        id: String,

        /// Metadata associated with the data, formatted for debugging.
        meta: String,
    },
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
    /// Creates a new `DataLoadError` indicating that data was not found.
    ///
    /// # Arguments
    ///
    /// * `id` - The identifier of the data that was not found.
    /// * `meta` - Metadata associated with the data, which will be formatted for debugging.
    ///
    /// # Returns
    ///
    /// A `DataLoadError` instance indicating that the specified data was not found.
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
