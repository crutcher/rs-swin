//! # Mathematical utilities.

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
    if exp == 1 {
        return Some(value);
    }
    if exp == 0 {
        return Some(1);
    }
    let (value, coeff) = if value >= 0 {
        (value, 1)
    } else if exp % 2 == 1 {
        (-value, -1)
    } else {
        // No real root for negative value with even exponent
        return None;
    };
    pos_maybe_iroot(value as usize, exp).map(|v| coeff * (v as isize))
}

/// Inner for `maybe_iroot`
#[inline(always)]
fn pos_maybe_iroot(
    value: usize,
    exp: usize,
) -> Option<usize> {
    if exp == 1 {
        return Some(value);
    }
    if exp == 0 {
        return Some(1);
    }
    match value {
        0 => Some(0),
        1 => Some(1),
        _ => {
            let exp = exp as u32;

            // Bisecting over the exclusive candidate range (lower, upper).
            let mut lower = 1usize;

            let mut upper = value.isqrt();
            if exp == 2 && value == upper.pow(2) {
                // Short-circuit for the builtin `isqrt` + squares.
                return Some(upper);
            }
            upper += 1;

            loop {
                if lower + 1 >= upper {
                    // No integer root found
                    return None;
                }
                let candidate = (lower + upper) / 2;

                match candidate.checked_pow(exp) {
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
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_maybe_iroot() {
        assert_eq!(maybe_iroot(0, 0), Some(1));
        assert_eq!(maybe_iroot(0, 2), Some(0));

        assert_eq!(maybe_iroot(4 * 4 * 4, 3), Some(4));

        assert_eq!(maybe_iroot(1, 2), Some(1));
        assert_eq!(maybe_iroot(4, 2), Some(2));
        assert_eq!(maybe_iroot(8, 3), Some(2));
        assert_eq!(maybe_iroot(27, 3), Some(3));

        // exp == 1 should return the value itself
        assert_eq!(maybe_iroot(5, 1), Some(5));
        assert_eq!(maybe_iroot(-7, 1), Some(-7));

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
