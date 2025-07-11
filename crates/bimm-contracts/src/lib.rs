#![warn(missing_docs)]
//! # BIMM Contracts

#![cfg_attr(all(feature = "nightly", test), feature(test))]
#[cfg(all(feature = "nightly", test))]
extern crate test;

extern crate core;

#[cfg(feature = "macros")]
pub use bimm_contracts_macros::shape_contract;

/// Evaluation Bindings.
pub mod bindings;
/// Shape Contracts.
pub mod contracts;
/// Dimension Expressions.
pub mod expressions;
/// Support Macros.
pub mod macros;
/// Mathematical utilities.
pub mod math;
/// Shape Argument for passing shapes in a type-safe manner.
pub mod shape_argument;

#[cfg(all(feature = "nightly", test))]
mod benchmarks;

pub use bindings::StackEnvironment;
pub use contracts::{DimMatcher, ShapeContract};
pub use expressions::DimExpr;
pub use shape_argument::ShapeArgument;
