use anyhow::{bail, Context};
use serde::de::DeserializeOwned;
use std::any::{Any, TypeId};
use std::fmt::Debug;

/// A wrapper type that can hold either a JSON value or a boxed value of any type.
pub enum ValueBox {
    /// Holds a JSON value.
    Value(serde_json::Value),

    /// Holds a boxed value of any type that implements `Any` and `Send`.
    Boxed(Box<dyn Any + 'static + Send>),
}


/// Checks if a type is boxable in a `ValueBox::Boxed`.
///
/// # Type Parameters
///
/// - `Outer`: The outer display type that will contain the boxed value.
/// - `Inner`: The inner type that is being checked.
///
/// # Returns
///
/// An `anyhow::Result<()>` indicating whether the type can be boxed.
pub fn try_boxable<T: 'static>() -> anyhow::Result<()> {
    let type_id = TypeId::of::<T>();
    let forbidden_types = [
        TypeId::of::<serde_json::Value>(),
        TypeId::of::<&'static str>(),
        TypeId::of::<&str>(),
        TypeId::of::<String>(),
        TypeId::of::<i32>(),
        TypeId::of::<i64>(),
        TypeId::of::<f32>(),
        TypeId::of::<f64>(),
        TypeId::of::<usize>(),
        TypeId::of::<isize>(),
        TypeId::of::<bool>(),
    ];
    if forbidden_types.contains(&type_id)  {
        let type_name = std::any::type_name::<T>();
        bail!(
            "Type `{type_name}` cannot be stored in ValueBox::Boxed; \
             use a ValueBox::Value instead."
        );
    }

    Ok(())
}

/// Checks if a type is boxable in a `ValueBox::Boxed`.
///
/// `panic!` wrapper for `try_boxable`.
pub fn check_boxable<T: 'static>() {
    match try_boxable::<T>() {
        Ok(_) => (),
        Err(e) => panic!("{e}"),
    }
}

impl Debug for ValueBox {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        match self {
            ValueBox::Value(value) => write!(f, "ValueBox::Value({value})"),
            ValueBox::Boxed(boxed) => write!(f, "ValueBox::Boxed({boxed:?})"),
        }
    }
}

impl ValueBox {
    /// Creates a new `ValueBox::Value` by serializing an object.
    pub fn from_serialize<T: serde::Serialize>(obj: T) -> anyhow::Result<Self> {
        serde_json::to_value(obj)
            .with_context(|| "Failed to serialize value")
            .map(ValueBox::Value)
    }

    /// Creates a new `ValueBox::Value` from a `serde_json::Value`.
    pub fn from_json_value(value: serde_json::Value) -> Self {
        ValueBox::Value(value)
    }

    /// Creates a new `ValueBox::Boxed` by boxing an object that implements `Any` and `Send`.
    pub fn boxing<T>(obj: T) -> Self
    where T: Any + Send + 'static,
    {
        Self::from_box(Box::new(obj))
    }

    /// Creates a new `ValueBox::Boxed` from a boxed object that implements `Any` and `Send`.
    pub fn from_box<T>(boxed: Box<T>) -> Self
    where T: Any + Send + 'static,
    {
        check_boxable::<T>();
        ValueBox::Boxed(boxed)
    }

    /// Returns true if the `ValueBox` contains a JSON value.
    pub fn is_value(&self) -> bool {
        matches!(self, ValueBox::Value(_))
    }

    /// Returns true if the `ValueBox` contains a boxed value.
    pub fn is_boxed(&self) -> bool {
        matches!(self, ValueBox::Boxed(_))
    }

    /// Unwraps the `ValueBox` and returns a reference to the contained JSON value.
    pub fn unwrap_value(&self) -> &serde_json::Value {
        if let ValueBox::Value(value) = self {
            value
        } else {
            panic!("ValueBox::unwrap_value() called on {self:?}");
        }
    }

    /// Unwraps and deserializes the contained JSON value into a specified type.
    ///
    /// # Type Parameters
    ///
    /// `T`: The type to deserialize the JSON value into. It must implement `DeserializeOwned`.
    ///
    /// # Panics
    ///
    /// If the `ValueBox` does not contain a JSON value.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<T>` where `T` is the deserialized type;
    /// if deserialization fails, it returns an error with context.
    pub fn deserialize_value<T>(&self) -> anyhow::Result<T>
    where
        T: DeserializeOwned + 'static,
    {
        let value = self.unwrap_value();
        serde_json::from_value(value.clone()).with_context(|| {
            format!(
                "ValueBox::deserialize_value::<{}>() failed on: {value}",
                std::any::type_name::<T>()
            )
        })
    }

    /// Unwraps the `ValueBox` and returns a reference to the contained boxed value.
    ///
    /// # Type Parameters
    ///
    /// `T`: The type to downcast the boxed value to.
    ///
    /// # Panics
    ///
    /// If the `ValueBox` does not contain a boxed value or if the downcast fails.
    ///
    /// # Returns
    ///
    /// An `anyhow::Result<&T>` where `T` is the down-cast type;
    /// if the downcast fails, it returns an error with context.
    pub fn unwrap_boxed<T>(&self) -> anyhow::Result<&T>
    where
        T: 'static,
    {
        if let ValueBox::Boxed(boxed) = self {
            boxed
                .downcast_ref::<T>()
                .with_context(|| "Failed to downcast boxed value")
        } else {
            panic!(
                "ValueBox::unwrap_boxed::<{}>() called on {self:?}",
                std::any::type_name::<T>()
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_try_boxable() {
        assert!(try_boxable::<serde_json::Value>().is_err());
        assert!(try_boxable::<&'static str>().is_err());
        assert!(try_boxable::<&str>().is_err());
        assert!(try_boxable::<String>().is_err());
        assert!(try_boxable::<i32>().is_err());
        assert!(try_boxable::<i64>().is_err());
        assert!(try_boxable::<f32>().is_err());
        assert!(try_boxable::<f64>().is_err());
        assert!(try_boxable::<usize>().is_err());
        assert!(try_boxable::<isize>().is_err());
        assert!(try_boxable::<bool>().is_err());

        assert!(try_boxable::<MyStruct>().is_ok());
    }

    #[test]
    fn test_from_json_value() -> anyhow::Result<()> {
        let json_value = serde_json::json!(["foo", "bar"]);
        let vb = ValueBox::from_json_value(json_value.clone());
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.deserialize_value::<Vec<String>>()?, vec!["foo", "bar"]);

        assert_eq!(format!("{vb:?}"), format!("ValueBox::Value({json_value})"));

        Ok(())
    }

    #[test]
    fn test_string_value() -> anyhow::Result<()> {
        let vb = ValueBox::from_serialize("abc")?;
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.deserialize_value::<String>()?, "abc");

        assert_eq!(
            vb.deserialize_value::<i32>().unwrap_err().to_string(),
            "ValueBox::deserialize_value::<i32>() failed on: \"abc\""
        );

        assert_eq!(vb.unwrap_value().as_str().unwrap(), "abc");

        Ok(())
    }

    #[test]
    #[should_panic(expected = r"called on Value")]
    fn test_unwrap_value_as_boxed() {
        let vb = ValueBox::from_serialize("abc").unwrap();
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        vb.unwrap_boxed::<MyStruct>().unwrap();
    }

    #[test]
    #[should_panic(expected = r"called on ValueBox::Boxed")]
    fn test_unwrap_boxed_as_value() {
        let my_struct = MyStruct {
            field1: "test".to_string(),
            field2: 123,
        };

        let vb = ValueBox::boxing(my_struct.clone());
        assert!(!vb.is_value());
        assert!(vb.is_boxed());

        vb.deserialize_value::<String>().unwrap();
    }

    #[test]
    fn test_int_value() -> anyhow::Result<()> {
        let vb = ValueBox::from_serialize(42_i32)?;
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.deserialize_value::<i32>()?, 42_i32);
        assert_eq!(vb.deserialize_value::<i64>()?, 42_i64);
        assert_eq!(vb.deserialize_value::<usize>()?, 42_usize);

        assert_eq!(vb.unwrap_value().as_i64().unwrap(), 42_i64);

        Ok(())
    }

    #[test]
    fn test_int_array() -> anyhow::Result<()> {
        let vb = ValueBox::from_serialize(vec![42_i32, 0_i32])?;
        assert!(vb.is_value());
        assert!(!vb.is_boxed());

        assert_eq!(vb.deserialize_value::<Vec<i32>>()?, vec![42_i32, 0_i32]);
        assert_eq!(vb.deserialize_value::<Vec<i64>>()?, vec![42_i64, 0_i64]);
        assert_eq!(
            vb.deserialize_value::<Vec<usize>>()?,
            vec![42_usize, 0_usize]
        );

        Ok(())
    }

    #[derive(serde::Serialize, serde::Deserialize, Debug, Clone, PartialEq, Eq)]
    pub struct MyStruct {
        pub field1: String,
        pub field2: i32,
    }

    #[test]
    fn test_boxed_value() -> anyhow::Result<()> {
        let my_struct = MyStruct {
            field1: "test".to_string(),
            field2: 123,
        };

        let vb = ValueBox::boxing(my_struct.clone());
        assert!(!vb.is_value());
        assert!(vb.is_boxed());

        assert_eq!(vb.unwrap_boxed::<MyStruct>()?, &my_struct);

        assert_eq!(
            vb.unwrap_boxed::<Vec<String>>().unwrap_err().to_string(),
            "Failed to downcast boxed value"
        );

        Ok(())
    }
}
