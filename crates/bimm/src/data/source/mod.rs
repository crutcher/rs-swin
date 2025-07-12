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

/// A filtered schedule source that applies a filter to the items in the schedule.
pub struct FilteredScheduleSource<M, F>
where
    M: DataLoadMetaDataItem,
    F: Fn(&M) -> bool,
{
    source: Box<dyn DataScheduleSource<M>>,
    filter: F,
}

impl<M, F> Debug for FilteredScheduleSource<M, F>
where
    M: DataLoadMetaDataItem,
    F: Fn(&M) -> bool,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("FilteredScheduleSource")
            .field("source", &self.source)
            .finish()
    }
}

impl<M, F> FilteredScheduleSource<M, F>
where
    M: DataLoadMetaDataItem,
    F: Fn(&M) -> bool,
{
    /// Creates a new `FilteredScheduleSource` with the given source and filter.
    pub fn new<S>(
        source: Box<dyn DataScheduleSource<M>>,
        filter: F,
    ) -> Self {
        Self { source, filter }
    }
}

impl<M, F> DataScheduleSource<M> for FilteredScheduleSource<M, F>
where
    M: DataLoadMetaDataItem,
    F: Fn(&M) -> bool,
{
    fn build_schedule(&self) -> Result<DataLoadSchedule<M>, DataLoadError> {
        let schedule = self.source.build_schedule()?;
        let filtered_items: Vec<M> = schedule
            .items
            .into_iter()
            .filter(|item| (self.filter)(item))
            .collect();
        Ok(DataLoadSchedule::from(filtered_items))
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
        let items = vec![2, 3, 5, 8];
        let source = FixedScheduleSource::from(items.clone());
        let filter = |&x: &usize| x % 2 == 0; // Filter even numbers
        let filtered_source = FilteredScheduleSource::new::<usize>(Box::new(source), filter);

        assert_eq!(
            filtered_source.build_schedule(),
            Ok(DataLoadSchedule::from(vec![2, 8]))
        );
    }
}
