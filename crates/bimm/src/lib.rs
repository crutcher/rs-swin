#![warn(missing_docs)]
//!# bimm - Burn Image Models

/// Test-only macro import.
#[cfg(test)]
#[allow(unused_imports)]
#[macro_use]
extern crate hamcrest;

/// The `compat` module provides compatibility layers for different versions of
/// Burn and its dependencies. It is not intended for public use and is
/// only available in the `bimm` crate.
#[allow(dead_code)]
pub(crate) mod compat;

/// Private module for internal use in the `bimm` crate.
#[cfg(test)]
#[allow(dead_code)]
pub(crate) mod testing;

/// Common low-level modules for adding layers and operations in Burn.
pub mod layers;

/// Common high-level modules for building models in Burn.
pub mod models;
pub mod utility;
