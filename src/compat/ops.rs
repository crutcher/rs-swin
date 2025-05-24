use crate::compat::dims::{canonicalize_dim, wrap_idx};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;
use std::iter::zip;

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

    _roll_dim(x, shift, dim)
}

/// Internal implementation of `roll_dim` that does not canonicalize dimensions or shifts.
pub fn _roll_dim<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    shift: usize,
    dim: usize,
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    let size = x.shape().dims[dim];
    assert!(shift < size);

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
    let dims = dims
        .iter()
        .map(|d| canonicalize_dim(*d, D, false))
        .collect::<Vec<_>>();
    let mut _shifts = Vec::with_capacity(shifts.len());
    for (shift, dim) in zip(shifts, dims.clone()) {
        let size = x.shape().dims[dim];
        if size == 0 {
            return x;
        }
        _shifts.push(wrap_idx(*shift, size));
    }

    _roll(x, &_shifts, &dims)
}

/// `roll` internal implementation.
#[inline(always)]
fn _roll<B: Backend, const D: usize, K>(
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

    let x = _roll_dim(x, shifts[0], dims[0]);
    if dims.len() == 1 {
        return x;
    }

    _roll(x, &shifts[1..], &dims[1..])
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
