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

/// A macro to run a block of code or an expression every nth time it is called.
///
/// This macro is useful for scenarios where you want to limit the execution
/// of code to every nth call, such as logging, sampling, or throttling operations.
///
/// ## Arguments:
///
/// - `$period`: A literal number indicating how often the code should run.
/// - `$code': An expression to be executed every nth time.
///
/// # Usage:
///  ```rust.no_run
///  // Run a block of code every 3rd call
///  run_every_nth!(3, {
///       println!("This will run every 3rd time.");
///       // Your code here
///  });
#[macro_export]
macro_rules! run_every_nth {
    ($period:literal, $code:expr) => {
        run_every_nth!(@internal $period, $code)
    };

    ($period:literal, $lock:block) => {
        run_every_nth!(@internal $period, $block)
    };

    (@internal $period:literal, $($tt:tt)*) => {{
        let period = $period;
        static COUNTER: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
        let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Always run on the first call.
        if count % period == 0 {
            $($tt)*
        }
        if count > period * 1000 {
            // Reset the counter to prevent overflow
            COUNTER.store(0, std::sync::atomic::Ordering::Relaxed);
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_every_nth_block() {
        let mut results = Vec::new();
        for i in 0..10 {
            run_every_nth!(2, {
                results.push(i);
            });
        }
        assert_eq!(results, vec![0, 2, 4, 6, 8]);
    }

    #[test]
    fn test_run_every_nth_expr() {
        let mut results = Vec::new();
        for i in 0..10 {
            run_every_nth!(2, results.push(i));
        }
        assert_eq!(results, vec![0, 2, 4, 6, 8]);
    }
}
