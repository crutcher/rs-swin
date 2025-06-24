/// A trait for looking up parameters in a stack-like environment.
pub trait StackMap<'a, V>
where
    V: Default + Copy,
{
    /// Looks up a value associated with the given key in the stack.
    ///
    /// ## Arguments
    ///
    /// * `key`: The key to look up.
    ///
    /// ## Returns
    ///
    /// An `Option<V>` containing the value if found, or `None` if not found.
    fn lookup(
        &self,
        key: &'a str,
    ) -> Option<V>;

    /// Exports the values for the given keys from the local bindings.
    ///
    /// ## Arguments
    ///
    /// * `keys`: An array of keys to look up in the local bindings.
    ///
    /// ## Returns
    ///
    /// An array of values corresponding to the keys. If a key is not found,
    ///
    /// ## Panics
    ///
    /// Panics if any key is not found in the local bindings.
    fn export_key_values<const K: usize>(
        &self,
        keys: &'a [&str; K],
    ) -> [V; K] {
        let mut values: [V; K] = [Default::default(); K];

        for i in 0..K {
            let key = keys[i];
            values[i] = match self.lookup(key) {
                Some(v) => v,
                None => panic!("No value for key \"{}\"", key),
            };
        }
        values
    }
}

/// Type alias for static/stack compatible bindings.
pub type StackEnvironment<'a> = &'a [(&'a str, usize)];

impl<'a> StackMap<'a, usize> for StackEnvironment<'a> {
    fn lookup(
        &self,
        key: &'a str,
    ) -> Option<usize> {
        self.iter()
            .find_map(|(k, v)| if k == &key { Some(*v) } else { None })
    }
}

/// A trait for mutable stack-like environments that allows inserting key-value pairs.
///
/// Provides no support for removing keys.
pub trait MutableStackMap<'a, V>: StackMap<'a, V>
where
    V: Default + Copy,
{
    /// Inserts a key-value pair into the stack.
    ///
    /// ## Arguments
    ///
    /// * `key`: The key to insert.
    /// * `value`: The value to associate with the key.
    ///
    /// ## Returns
    ///
    /// This method does not return a value,
    /// but modifies the stack by inserting the key-value pair.
    fn bind(
        &mut self,
        key: &'a str,
        value: V,
    );
}

/// A mutable stack environment, backed by a static stack environment.
pub struct MutableStackEnvironment<'a> {
    pub backing: StackEnvironment<'a>,
    pub updates: Vec<(&'a str, usize)>,
}

impl<'a> StackMap<'a, usize> for MutableStackEnvironment<'a> {
    fn lookup(
        &self,
        key: &str,
    ) -> Option<usize> {
        (&self.updates.as_slice()).lookup(key).or_else(|| {
            // If not found in local, check the static bindings
            self.backing.lookup(key)
        })
    }
}

impl<'a> MutableStackMap<'a, usize> for MutableStackEnvironment<'a> {
    fn bind(
        &mut self,
        key: &'a str,
        value: usize,
    ) {
        self.updates.push((key, value))
    }
}

impl<'a> MutableStackEnvironment<'a> {
    /// Creates a new `MutableStackEnvironment` with the given backing bindings.
    pub fn new(bindings: StackEnvironment<'a>) -> Self {
        MutableStackEnvironment {
            backing: bindings,
            updates: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_stack_bindings() {
        {
            // stack bindings.
            let env: StackEnvironment = &[("a", 1), ("b", 2)];

            assert_eq!(env.lookup("a"), Some(1));
            assert_eq!(env.lookup("b"), Some(2));
            assert_eq!(env.lookup("c"), None);
        }
        {
            // static bindings.
            static ENV: StackEnvironment = &[("a", 1), ("b", 2)];

            assert_eq!(ENV.lookup("a"), Some(1));
            assert_eq!(ENV.lookup("b"), Some(2));
            assert_eq!(ENV.lookup("c"), None);
        }
    }

    #[test]
    fn test_mutable_stack_bindings() {
        let mut env = MutableStackEnvironment::new(&[("a", 1), ("b", 2)]);

        assert_eq!(env.lookup("a"), Some(1));
        assert_eq!(env.lookup("b"), Some(2));
        assert_eq!(env.lookup("c"), None);

        env.bind("c", 3);
        assert_eq!(env.lookup("c"), Some(3));

        // Exporting values
        let keys = ["a", "b", "c"];
        let values = env.export_key_values(&keys);
        assert_eq!(values, [1, 2, 3]);
    }

    #[should_panic(expected = "No value for key \"d\"")]
    #[test]
    fn test_export_key_values_panic() {
        let env = MutableStackEnvironment::new(&[("a", 1), ("b", 2)]);
        let keys = ["a", "b", "d"]; // 'd' does not exist
        env.export_key_values(&keys); // This should panic
    }
}
