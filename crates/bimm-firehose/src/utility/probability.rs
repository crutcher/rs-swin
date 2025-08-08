/// Validate a probability range.
///
/// ## Arguments
///
/// - `prob`: the prob to check.
///
/// ## Returns
///
/// An `anyhow::Result<f64>`
pub fn try_probability(prob: f64) -> anyhow::Result<f64> {
    if !(0.0..=1.0).contains(&prob) {
        return Err(anyhow::anyhow!("probability must be in [0.0, 1.0]: {prob}"));
    }
    Ok(prob)
}

/// Expect a probability range.
///
/// ## Arguments
///
/// - `prob`: the prob to check.
///
/// ## Returns
///
/// An `f64`.
///
/// ## Panics
///
/// On range error.
pub fn expect_probability(prob: f64) -> f64 {
    try_probability(prob).unwrap()
}

#[cfg(test)]
mod tests {
    use crate::utility::probability::{expect_probability, try_probability};

    #[test]
    fn test_probability() {
        for v in [0.0, 0.5, 1.0].iter() {
            assert!(try_probability(*v).is_ok());
            assert!(expect_probability(*v) == *v);
        }

        assert!(try_probability(-1.0).is_err());
        assert!(try_probability(2.0).is_err());
    }

    #[should_panic]
    #[test]
    fn test_probability_panic() {
        expect_probability(-1.0);
    }
}
