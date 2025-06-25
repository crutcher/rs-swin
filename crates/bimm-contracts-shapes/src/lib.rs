extern crate core;

pub mod bindings;
pub mod contracts;
pub mod expressions;
pub mod math;

pub use bindings::StackEnvironment;
pub use contracts::{DimMatcher, ShapeContract};
pub use expressions::DimExpr;
