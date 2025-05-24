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
pub fn canonicalize_dim(
    rank: usize,
    idx: isize,
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
    rank: usize,
    dims: &[isize],
    wrap_scalar: bool,
) -> Vec<usize> {
    dims.iter()
        .map(|&idx| canonicalize_dim(rank, idx, wrap_scalar))
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
pub fn is_valid_permutation(
    rank: usize,
    perm: &[usize],
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
pub fn canonicalize_permutation(
    rank: usize,
    perm: &[isize],
) -> Vec<usize> {
    let _perm = canonicalize_dims(rank, perm, false);
    if !is_valid_permutation(rank, &_perm) {
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
    fn test_canonicalize_dim() {
        for idx in 0..3 {
            let wrap_scalar = false;
            assert_eq!(canonicalize_dim(3, idx, wrap_scalar), idx as usize);
            assert_eq!(
                canonicalize_dim(3, -(idx + 1), wrap_scalar),
                (3 - (idx + 1)) as usize
            );
        }

        let wrap_scalar = true;
        assert_eq!(canonicalize_dim(0, 0, wrap_scalar), 0);
        assert_eq!(canonicalize_dim(0, -1, wrap_scalar), 0);
    }

    #[test]
    #[should_panic = "Dimension specified as 0 but tensor has no dimensions"]
    fn test_canonicalize_error_no_dims() {
        canonicalize_dim(0, 0, false);
    }

    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got 3)"]
    fn test_canonicalize_error_too_big() {
        canonicalize_dim(3, 3, false);
    }
    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got -4)"]
    fn test_canonicalize_error_too_small() {
        canonicalize_dim(3, -4, false);
    }

    #[test]
    fn test_canonicalize_dims() {
        let dims = vec![0, 1, 2];
        let wrap_scalar = false;
        assert_eq!(canonicalize_dims(3, &dims, wrap_scalar), vec![0, 1, 2]);
        assert_eq!(canonicalize_dims(3, &[-1, -2], wrap_scalar), vec![2, 1]);
    }

    #[test]
    fn test_is_valid_permutation() {
        let rank = 3;
        let valid_perm = vec![2, 0, 1];
        let invalid_perm = vec![2, 0, 3];

        assert!(is_valid_permutation(rank, &valid_perm));
        assert!(!is_valid_permutation(rank, &invalid_perm));
    }

    #[test]
    #[should_panic = "Invalid permutation: expected a permutation of length 3, but got [0, 1]"]
    fn test_canonicalize_permutation_error_too_short() {
        let rank = 3;
        let perm = vec![0, 1];
        canonicalize_permutation(rank, &perm);
    }

    #[test]
    #[should_panic = "Dimension out of range (expected to be in range of [-3, 2], but got 3)"]
    fn test_canonicalize_permutation_error_too_long() {
        let rank = 3;
        let perm = vec![0, 1, 2, 3];
        canonicalize_permutation(rank, &perm);
    }

    #[test]
    #[should_panic = "Invalid permutation: expected a permutation of length 3, but got [0, 1, 1]"]
    fn test_canonicalize_permutation_error_duplicate() {
        let rank = 3;
        let perm = vec![0, 1, 1];
        canonicalize_permutation(rank, &perm);
    }

    #[test]
    fn test_canonicalize_permutation() {
        let rank = 3;
        let perm = vec![-1, -3, 1];
        assert_eq!(canonicalize_permutation(rank, &perm), vec![2, 0, 1]);
    }
}
