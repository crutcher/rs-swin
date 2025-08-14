//! Support macros.

/// A macro to run a block of code or an expression every nth time it is called.
///
/// Runs the first 10 times, then doubles the period on each subsequent call,
/// until it reaches the specified period, after which it continues to run at that period.
///
/// This macro is useful for scenarios where you want to limit the execution
/// of code to every nth call, such as logging, sampling, or throttling operations.
///
/// ## Arguments:
///
/// - `$period`: [optional; default=1000] the period.
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
        run_every_nth!(@internal 1000, $code)
    };

    ($lock:block) => {
        run_every_nth!(@internal 1000, $block)
    };

    ($period:literal, $code:expr) => {
        run_every_nth!(@internal $period, $code)
    };

    ($period:literal, $lock:block) => {
        run_every_nth!(@internal $period, $block)
    };

    (@internal $period:literal, $($tt:tt)*) => {{
        if {
            use core::sync::atomic::AtomicUsize;
            use core::sync::atomic::Ordering;

            static PERIOD: AtomicUsize = AtomicUsize::new(1);
            static COUNTER: AtomicUsize = AtomicUsize::new(0);

            let effective_period = PERIOD.load(Ordering::Relaxed);
            let count = COUNTER.fetch_add(1, Ordering::Relaxed);

            if effective_period == 1 && count < 10 {
                true

            } else if (count % effective_period) == 0 {
                // Double the period, but do not exceed the specified maximum period.
                if effective_period < $period {
                    PERIOD.store(
                        (2 * effective_period).clamp(1, $period),
                        Ordering::Relaxed,
                    );
                }
                // Reset the counter when we alter the period;
                // or periodically reset it to avoid overflow.
                if effective_period < $period || count > $period * 100 {
                    COUNTER.store(1, Ordering::Relaxed);
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
    use alloc::vec;
    use alloc::vec::Vec;

    #[test]
    fn test_run_every_nth() {
        let expected = vec![
            0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12, 16, 24, 40, 72, 136, 264, 520, 1032, 2032,
        ];

        // Block.
        {
            let mut results = Vec::new();
            for i in 0..2500 {
                run_every_nth!({
                    results.push(i);
                });
            }
            assert_eq!(&results, &expected);
        }

        // Expression.
        {
            let mut results = Vec::new();
            for i in 0..2500 {
                run_every_nth!(results.push(i));
            }
            assert_eq!(&results, &expected);
        }
    }
}
