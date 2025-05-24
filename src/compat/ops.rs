use crate::compat::dims::{canonicalize_dim, wrap_idx};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;

/// Roll operation along a specific dimension.
///
/// ## Parameters
///
/// - `x`: The input tensor.
/// - `shift`: The number of positions to shift; supports negative values and wraps around.
/// - `dim`: The dimension to roll; supports negative indexing.
///
/// ## Returns
///
/// A new tensor with the specified dimension rolled by the given shift amount.
pub fn roll_dim<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    shift: isize,
    dim: isize,
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    let dim = canonicalize_dim(dim, D, false);
    let size = x.shape().dims[dim];
    let shift = wrap_idx(shift, size);

    _unchecked_roll_dim(x, shift, dim)
}

/// Internal implementation of `roll_dim` that does not canonicalize dimensions or shifts.
#[inline(always)]
pub fn _unchecked_roll_dim<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    shift: usize,
    dim: usize,
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    // By contract: assert!(shift < size);
    let size = x.shape().dims[dim];
    if size == 0 || shift == 0 {
        return x;
    }

    let mut parts = x.split_with_sizes(vec![shift, size - shift], dim);
    parts.rotate_right(1);

    Tensor::cat(parts, dim)
}

/// Roll operation.
///
/// Note: unlike ``pytorch``, `dims` and `shifts` must have the same length.
///
/// A given `dim` may be rolled multiple times, and the shifts will be applied sequentially.
///
/// ## Parameters
///
/// - `x`: The input tensor.
/// - `shifts`: A slice of shifts corresponding to each dimension; supports negative values and wraps around.
/// - `dims`: A slice of dimensions to roll; supports negative indexing.
///
/// ## Returns
///
/// A new tensor with the specified dimensions rolled by the given shifts.
pub fn roll<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    shifts: &[isize],
    dims: &[isize],
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    assert_eq!(
        dims.len(),
        shifts.len(),
        "Dimensions and shifts must align; found {} dims and {} shifts",
        dims.len(),
        shifts.len()
    );

    // Canonicalize dimensions and shifts before any copies or modifications.
    // This is duplicated check logic, but enforces no-copy on no-op.
    let cannon_dims: Vec<usize> = dims
        .iter()
        .map(|d| canonicalize_dim(*d, D, false))
        .collect();

    // Do this after we've checked the validity of `dims` and `shifts`.
    if x.shape().num_elements() == 0 {
        // If the tensor is empty, return it as is.
        return x;
    }

    // All relative shifts are legal; we just need them wrapped.
    let cannon_shifts: Vec<usize> = shifts
        .iter()
        .zip(cannon_dims.iter())
        .map(|(shift, &dim)| {
            let size = x.shape().dims[dim];
            wrap_idx(*shift, size)
        })
        .collect();

    _unchecked_roll(x, &cannon_shifts, &cannon_dims)
}

/// `roll` internal implementation.
#[inline(always)]
fn _unchecked_roll<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    shifts: &[usize],
    dims: &[usize],
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    if dims.is_empty() {
        return x;
    }

    let x = _unchecked_roll_dim(x, shifts[0], dims[0]);
    if dims.len() == 1 {
        return x;
    }

    _unchecked_roll(x, &shifts[1..], &dims[1..])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::{Int, TensorData};
    use burn::tensor::Tensor;

    #[test]
    fn test_roll_empty() {
        let device = Default::default();
        let input: Tensor<NdArray, 2, Int> = Tensor::zeros([12, 0], &device);

        // Rolling an empty tensor should return the same empty tensor
        roll(input.clone(), &[1, 2], &[0, 1])
            .to_data()
            .assert_eq(&input.to_data(), false);
    }

    #[test]
    fn test_roll() {
        let device = Default::default();
        let input: Tensor<NdArray, 2, Int> = Tensor::arange(0..6, &device).reshape::<2, _>([2, 3]);

        // No-op shift:
        roll(input.clone(), &[0, 0], &[0, 1])
            .to_data()
            .assert_eq(&input.clone().to_data(), false);

        roll(input.clone(), &[1, -1], &[0, 1])
            .to_data()
            .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);

        roll(input.clone(), &[2 * 32 + 1, 3 * (-400) - 1], &[0, 1])
            .to_data()
            .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);
    }
}
