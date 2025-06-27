//! # BIMM Contracts

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
/// Runs at a doubling rate (1, 2, 4, ...), until it reaches the specified period;
/// then continues to run at that period.
///
/// This macro is useful for scenarios where you want to limit the execution
/// of code to every nth call, such as logging, sampling, or throttling operations.
///
/// ## Arguments:
///
/// - `$period`: A literal number indicating how often the code should run;
///   optional, defaults to 100.
/// - `$code`: An expression to be executed every nth time.
///
/// # Usage:
/// ```rust.no_run
///  use bimm_contracts::run_every_nth;
///
///  // Run a block of code every 3rd call
/// run_every_nth!(3, {
///       println!("This will run every 3rd time.");
///       // Your code here
///  });
#[macro_export]
macro_rules! run_every_nth {
    ($code:expr) => {
        run_every_nth!(@internal 100, $code)
    };

    ($lock:block) => {
        run_every_nth!(@internal 100, $block)
    };

    ($period:literal, $code:expr) => {
        run_every_nth!(@internal $period, $code)
    };

    ($period:literal, $lock:block) => {
        run_every_nth!(@internal $period, $block)
    };

    (@internal $period:literal, $($tt:tt)*) => {{
        if {
            static PERIOD: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(1);
            static COUNTER: std::sync::atomic::AtomicUsize =
                std::sync::atomic::AtomicUsize::new(0);

            let effective_period = PERIOD.load(std::sync::atomic::Ordering::Relaxed);
            let count = COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);

            if (count % effective_period) == 0 {
                // Double the period, but do not exceed the specified maximum period.
                if effective_period < $period {
                    PERIOD.store(
                        (effective_period * 2).clamp(1, $period),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }

                // Reset the counter to prevent overflow
                if count > $period * 1000 {
                    COUNTER.store(0, std::sync::atomic::Ordering::Relaxed);
                }

                true
            } else {
                false
            }
        } {
            $($tt)*
        }
    }};
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_run_every_nth_block() {
        let mut results = Vec::new();
        for i in 0..350 {
            run_every_nth!({
                results.push(i);
            });
        }
        assert_eq!(results, vec![0, 2, 4, 8, 16, 32, 64, 100, 200, 300]);
    }

    #[test]
    fn test_run_every_nth_expr() {
        let mut results = Vec::new();
        for i in 0..350 {
            run_every_nth!(results.push(i));
        }
        assert_eq!(results, vec![0, 2, 4, 8, 16, 32, 64, 100, 200, 300]);
    }
}
