use crate::data::error::DataLoadError;
use crate::data::plan::{DataLoadMetaDataItem, DataLoadSchedule};
use std::fmt::Debug;

/// Data-pipeline trait for schedule sources.
pub trait DataScheduleSource<M>: Debug
where
    M: DataLoadMetaDataItem,
{
    /// Builds a schedule from the source.
    fn build_schedule(&self) -> Result<DataLoadSchedule<M>, DataLoadError>;
}

/// A pre-built schedule source that provides a fixed schedule.
#[derive(Debug)]
pub struct FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    schedule: DataLoadSchedule<M>,
}

impl<M> From<DataLoadSchedule<M>> for FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn from(schedule: DataLoadSchedule<M>) -> Self {
        Self { schedule }
    }
}

impl<M> From<Vec<M>> for FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn from(schedule: Vec<M>) -> Self {
        Self::from(DataLoadSchedule::from(schedule))
    }
}

impl<M> DataScheduleSource<M> for FixedScheduleSource<M>
where
    M: DataLoadMetaDataItem,
{
    fn build_schedule(&self) -> Result<DataLoadSchedule<M>, DataLoadError> {
        Ok(self.schedule.clone())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_fixed_schedule_source() {
        let items = vec![2, 3, 5];
        let source = FixedScheduleSource::from(items.clone());

        assert_eq!(
            source.build_schedule(),
            Ok(DataLoadSchedule::from(items.clone()))
        );
    }

    #[test]
    fn test_filter() {
        todo!("Implement generic filter_map for sources");
    }
}
