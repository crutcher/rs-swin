use burn::serde::Serialize;
use serde::Deserialize;
use std::fmt::Debug;
use std::hash::Hash;

/// A fixed schedule for loading data items.
///
/// The type `M` must implement the `ScheduleItem` trait, which requires it to be
/// Debug, Serialize, Sync, and Send safe.
///
/// ## Motivation
///
/// This structure is designed to hold the intermediate schedule for a data pipeline
/// load operation; in a format which can be serialized and deserialized;
/// and can be transformed into modified schedules without forcing a data load.
///
/// As such, it is a thin wrapper around a `Vec<M>`, where `M` carries complex
/// constraints; but the expectation is that additional metadata will be added
/// to the schedule in the future (such as block size, or item load cost estimates).
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(bound = "M: DataLoadMetaDataItem")]
pub struct DataLoadSchedule<M>
where
    M: DataLoadMetaDataItem,
{
    pub items: Vec<M>,
}

/// Support super-trait for `DataLoadSchedule` items.
///
/// This trait exists to simplify the requirements for items in a `DataLoadSchedule`;
/// blanket implementations of `ScheduleItem` are provided for any type that meets
/// the requirements.
///
/// ## Trait Requirements
///
/// - `Debug`: the schedule must be displayable.
/// - `Clone`: the schedule must be cloneable.
/// - `PartialEq` and `Eq`: the schedule must be comparable for equality.
/// - `Send` and `Sync`: the schedule must be thread-safe.
/// - `Serialize` and `Deserialize`: the schedule must be serializable and deserializable.
pub trait DataLoadMetaDataItem:
    Debug + Clone + Hash + PartialEq + Eq + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
}

/// Blanket implementation of `ScheduleItem` for any type that meets the requirements.
impl<T> DataLoadMetaDataItem for T where
    T: Debug + Clone + Hash + PartialEq + Eq + Send + Sync + Serialize + for<'de> Deserialize<'de>
{
}

impl<M> Debug for DataLoadSchedule<M>
where
    M: DataLoadMetaDataItem,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        if f.alternate() {
            f.write_str("DataLoadSchedule {\n")?;
            for (idx, item) in self.items.iter().enumerate() {
                f.write_str(&format!("{idx}: {item:?},\n"))?;
            }
            f.write_str("}")
        } else {
            f.debug_struct("DataLoadSchedule")
                .field("items", &self.items)
                .finish()
        }
    }
}

impl<M> DataLoadSchedule<M>
where
    M: DataLoadMetaDataItem,
{
    /// Creates a new empty `DataLoadSchedule`.
    pub fn new() -> Self {
        From::from(Vec::new())
    }

    /// Gets the number of items in the schedule.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// Checks if the schedule is empty.
    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }
}

impl<M> From<Vec<M>> for DataLoadSchedule<M>
where
    M: DataLoadMetaDataItem,
{
    fn from(items: Vec<M>) -> Self {
        Self { items }
    }
}

impl<M> Default for DataLoadSchedule<M>
where
    M: DataLoadMetaDataItem,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_index_schedule() {
        let vec = vec![1, 2, 3];
        let mut schedule = DataLoadSchedule::from(vec.clone());

        {
            // Check the Into trait implementation.
            let alt: DataLoadSchedule<i32> = vec.clone().into();
            assert_eq!(alt, schedule);
        }

        schedule.items.push(1);
        assert_eq!(schedule.len(), 4);
        assert_eq!(schedule.items.get(3), Some(&1));

        // Check the Debug trait implementation.
        assert_eq!(
            format!("{schedule:?}"),
            "DataLoadSchedule { items: [1, 2, 3, 1] }"
        );
        assert_eq!(
            format!("{schedule:#?}"),
            "\
DataLoadSchedule {
0: 1,
1: 2,
2: 3,
3: 1,
}"
        );

        // Check the json/serde implementation.
        {
            let json = serde_json::to_string(&schedule).unwrap();
            assert_eq!(json, "{\"items\":[1,2,3,1]}");
            let obj = serde_json::from_str::<DataLoadSchedule<i32>>(&json).unwrap();

            assert_eq!(obj, schedule);
        }
    }

    #[derive(Debug, Clone, Hash, Serialize, Deserialize, PartialEq, Eq)]
    pub struct ExampleItem {
        path: String,
    }

    #[test]
    fn test_struct_schedule() {
        let item = ExampleItem {
            path: String::from("/example/path"),
        };
        let mut schedule = DataLoadSchedule::new();
        schedule.items.push(item.clone());

        assert_eq!(schedule.len(), 1);
        assert_eq!(schedule.items.first(), Some(&item));
        assert!(!schedule.is_empty());

        // Check the Debug trait implementation.
        assert_eq!(
            format!("{schedule:#?}"),
            "\
DataLoadSchedule {
0: ExampleItem { path: \"/example/path\" },
}"
        );

        // Check the json/serde implementation.
        {
            let json = serde_json::to_string(&schedule).unwrap();
            assert_eq!(json, "{\"items\":[{\"path\":\"/example/path\"}]}");
            let obj = serde_json::from_str::<DataLoadSchedule<ExampleItem>>(&json).unwrap();

            assert_eq!(obj, schedule);
        }
    }
}
