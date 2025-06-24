/// Find the exact integer nth root if it exists.
///
/// ## Arguments
///
/// - `value` - the value.
/// - `exp` - the exponent.
///
/// ## Returns
///
/// Either the exact root (positive if possible) or None.
pub fn maybe_iroot(
    value: isize,
    exp: usize,
) -> Option<isize> {
    match value {
        0 => {
            if exp == 0 {
                Some(1)
            } else {
                Some(0)
            }
        }
        1 => Some(1),
        v if v > 1 => {
            let n = exp as u32;

            // Bin search for the integer root.
            let mut lower = 1isize;
            let mut upper = value.isqrt() + 1;
            loop {
                if lower + 1 >= upper {
                    // No integer root found
                    return None;
                }
                let candidate = (lower + upper) / 2;

                match candidate.checked_pow(n) {
                    None => {
                        // The candidate overflows; reduce the upper bound.
                        upper = candidate - 1;
                    }
                    Some(pow) => {
                        if pow == value {
                            // Found the exact root
                            return Some(candidate);
                        } else if pow > value {
                            // The candidate is too high, reduce the upper bound
                            upper = candidate;
                        } else {
                            // The candidate is too low, increase the lower bound
                            lower = candidate;
                        }
                    }
                }
            }
        }
        v => {
            // Negative values: only possible for odd exponents
            if exp % 2 == 1 {
                // For odd n, (-x)^n = -(x^n), so we can find the positive root and negate
                maybe_iroot(-v, exp).map(|root| -root)
            } else {
                None // No real root for negative value with even exponent
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_int_root() {
        assert_eq!(maybe_iroot(0, 0), Some(1));
        assert_eq!(maybe_iroot(0, 2), Some(0));

        assert_eq!(maybe_iroot(4 * 4 * 4, 3), Some(4));

        assert_eq!(maybe_iroot(1, 2), Some(1));
        assert_eq!(maybe_iroot(4, 2), Some(2));
        assert_eq!(maybe_iroot(8, 3), Some(2));
        assert_eq!(maybe_iroot(27, 3), Some(3));

        assert_eq!(maybe_iroot(-8, 3), Some(-2));
        assert_eq!(maybe_iroot(-16, 4), None);
        assert_eq!(maybe_iroot(16, 4), Some(2));
        assert_eq!(maybe_iroot(-1, 3), Some(-1));
        assert_eq!(maybe_iroot(-1, 2), None);

        // Overflow case
        let too_big = isize::MAX.isqrt() + 1;
        assert_eq!(maybe_iroot(too_big, 2), None);
        assert_eq!(maybe_iroot(too_big, 6), None);
    }
}
