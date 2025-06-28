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
                        (2 * effective_period).clamp(1, $period),
                        std::sync::atomic::Ordering::Relaxed,
                    );
                }
                // Reset the counter when we alter the period;
                // or periodically reset it to avoid overflow.
                if effective_period < $period || count > $period * 100 {
                    COUNTER.store(1, std::sync::atomic::Ordering::Relaxed);
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

    #[test]
    fn test_run_every_nth() {
        let expected = vec![0, 2, 6, 14, 30, 62, 126, 254, 510, 1022, 2022];

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
