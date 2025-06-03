/// Canonicalizes and bounds checks a dimension index.
///
/// ## Arguments
///
/// * `rank` - The rank of the tensor.
/// * `idx` - The dimension index to canonicalize.
/// * `wrap_scalar` - If true, pretend scalars have rank=1.
///
/// ## Returns
///
/// The canonicalized dimension index.
///
/// ## Panics
///
/// * If `wrap_scalar` is false and the tensor has no dimensions.
/// * If the dimension index is out of range.
#[must_use]
pub fn canonicalize_dim(
    idx: isize,
    rank: usize,
    wrap_scalar: bool,
) -> usize {
    let rank = if rank > 0 {
        rank
    } else {
        if !wrap_scalar {
            panic!("Dimension specified as {idx} but tensor has no dimensions");
        }
        1
    };

    if idx >= 0 && (idx as usize) < rank {
        return idx as usize;
    }

    let _idx = if idx < 0 { idx + rank as isize } else { idx };

    if _idx < 0 || (_idx as usize) >= rank {
        panic!(
            "Dimension out of range (expected to be in range of [{}, {}], but got {})",
            -(rank as isize),
            rank - 1,
            idx
        );
    }

    _idx as usize
}

/// Wraps a dimension index to be within the bounds of the dimension size.
///
/// ## Arguments
///
/// * `idx` - The dimension index to wrap.
/// * `size` - The size of the dimension.
///
/// ## Returns
///
/// The positive wrapped dimension index.
#[must_use]
pub fn wrap_idx(
    idx: isize,
    size: usize,
) -> usize {
    if size == 0 {
        return 0; // Avoid modulo by zero
    }
    let wrapped = idx.rem_euclid(size as isize);
    if wrapped < 0 {
        (wrapped + size as isize) as usize
    } else {
        wrapped as usize
    }
}

/// Canonicalizes and bounds checks a list of dimension indices.
///
/// ## Arguments
///
/// * `rank` - The rank of the tensor.
/// * `dims` - The dimension indices to canonicalize.
/// * `wrap_scalar` - If true, pretend scalars have rank=1.
///
/// ## Returns
///
/// A vector of canonicalized dimension indices.
pub fn canonicalize_dims(
    dims: &[isize],
    rank: usize,
    wrap_scalar: bool,
) -> Vec<usize> {
    dims.iter()
        .map(|&idx| canonicalize_dim(idx, rank, wrap_scalar))
        .collect()
}

/// Checks if a permutation is valid.
///
/// ## Arguments
///
/// * `rank` - The rank of the tensor.
/// * `perm` - The permutation to check.
///
/// ## Returns
///
/// True if the permutation is valid, false otherwise.
#[must_use]
pub fn is_valid_permutation(
    perm: &[usize],
    rank: usize,
) -> bool {
    if perm.len() != rank {
        return false;
    }

    let mut seen = vec![false; rank];
    for &idx in perm {
        if idx >= rank || seen[idx] {
            return false;
        }
        seen[idx] = true;
    }
    true
}

/// Converts a permutation to a canonical form.
///
/// ## Arguments
///
/// * `rank` - The rank of the tensor.
/// * `perm` - The permutation to convert.
///
/// ## Returns
///
/// A vector of canonicalized dimension indices.
///
/// ## Panics
///
/// * If the permutation is invalid.
#[must_use]
pub fn canonicalize_permutation(
    perm: &[isize],
    rank: usize,
) -> Vec<usize> {
    let _perm = canonicalize_dims(perm, rank, false);
    if !is_valid_permutation(&_perm, rank) {
        panic!(
            "Invalid permutation: expected a permutation of length {}, but got {:?}",
            rank, perm
        );
    }
    _perm
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_wrap_idx() {
        for idx in 0..3 {
            assert_eq!(wrap_idx(idx, 3), idx as usize);
            assert_eq!(wrap_idx(idx + 3, 3), idx as usize);
            assert_eq!(wrap_idx(idx + 2 * 3, 3), idx as usize);
            assert_eq!(wrap_idx(idx - 3, 3), idx as usize);
            assert_eq!(wrap_idx(idx - 2 * 3, 3), idx as usize);
        }
    }

    #[test]
    fn test_canonicalize_dim() {
        for idx in 0..3 {
            let wrap_scalar = false;
            assert_eq!(canonicalize_dim(idx, 3, wrap_scalar), idx as usize);
            assert_eq!(
                canonicalize_dim(-(idx + 1), 3, wrap_scalar),
                (3 - (idx + 1)) as usize
            );
        }

        let wrap_scalar = true;
        assert_eq!(canonicalize_dim(0, 0, wrap_scalar), 0);
        assert_eq!(canonicalize_dim(-1, 0, wrap_scalar), 0);
    }

    #[test]
    #[should_panic = "Dimension specified as 0 but tensor has no dimensions"]
    fn test_canonicalize_error_no_dims() {
        let _d = canonicalize_dim(0, 0, false);
    }

    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got 3)"]
    fn test_canonicalize_error_too_big() {
        let _d = canonicalize_dim(3, 3, false);
    }
    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got -4)"]
    fn test_canonicalize_error_too_small() {
        let _d = canonicalize_dim(-4, 3, false);
    }

    #[test]
    fn test_canonicalize_dims() {
        let dims = vec![0, 1, 2];
        let wrap_scalar = false;
        assert_eq!(canonicalize_dims(&dims, 3, wrap_scalar), vec![0, 1, 2]);
        assert_eq!(canonicalize_dims(&[-1, -2], 3, wrap_scalar), vec![2, 1]);
    }

    #[test]
    fn test_is_valid_permutation() {
        let rank = 3;
        let valid_perm = vec![2, 0, 1];
        let invalid_perm = vec![2, 0, 3];

        assert!(is_valid_permutation(&valid_perm, rank));
        assert!(!is_valid_permutation(&invalid_perm, rank));
    }

    #[test]
    #[should_panic = "Invalid permutation: expected a permutation of length 3, but got [0, 1]"]
    fn test_canonicalize_permutation_error_too_short() {
        let rank = 3;
        let perm = vec![0, 1];
        let _p = canonicalize_permutation(&perm, rank);
    }

    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got 3)"]
    fn test_canonicalize_permutation_error_too_long() {
        let rank = 3;
        let perm = vec![0, 1, 2, 3];
        let _p = canonicalize_permutation(&perm, rank);
    }

    #[test]
    #[should_panic = "Invalid permutation: expected a permutation of length 3, but got [0, 1, 1]"]
    fn test_canonicalize_permutation_error_duplicate() {
        let rank = 3;
        let perm = vec![0, 1, 1];
        let _p = canonicalize_permutation(&perm, rank);
    }

    #[test]
    fn test_canonicalize_permutation() {
        let rank = 3;
        let perm = vec![-1, -3, 1];
        assert_eq!(canonicalize_permutation(&perm, rank), vec![2, 0, 1]);
    }
}
