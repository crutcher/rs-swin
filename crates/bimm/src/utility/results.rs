//! # Result Utilities
//!
//! Methods for [`std::result::Result`] manipulation.

use std::fmt::Debug;

/// Unwraps Result, or Panics.
///
/// Unlike the `.unwrap()` method, this does not add a prefix about
/// `.unwrap()`
///
/// Useful for building ``expect_<func>(...) -> T`` variants of
/// ``try_<func>(...) -> Result<T, E>`` methods.
pub fn expect_unwrap<T, E>(result: core::result::Result<T, E>) -> T
where
    E: Debug,
{
    match result {
        Ok(t) => t,
        Err(e) => panic!("{:?}", e),
    }
}
