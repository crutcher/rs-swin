#![cfg_attr(all(feature = "nightly", test), feature(test))]
#[cfg(all(feature = "nightly", test))]
extern crate test;

extern crate core;

pub mod bindings;
pub mod contracts;
pub mod expressions;
pub mod math;
pub mod shape_argument;

pub use bindings::StackEnvironment;
pub use contracts::{DimMatcher, ShapeContract};
pub use expressions::DimExpr;
pub use shape_argument::ShapeArgument;
