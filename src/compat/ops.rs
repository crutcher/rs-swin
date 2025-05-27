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

    if size == 0 || shift == 0 {
        return x;
    }

    _unchecked_roll_dim(x, shift, dim)
}

/// Contract for the `_unchecked_roll_dim` operation.
///
/// ## Parameters
///
/// - `shift`: The number of positions to shift; must be (0 < shift < size).
/// - `dim`: The dimension to roll; must be a valid index for the tensor's shape.
/// - `size`: The size of the dimension to roll; must be greater than 0.
///
/// ## Panics
///
/// Panics if the contract conditions are not met.
fn _unchecked_roll_dim_contract(
    shift: usize,
    dim: usize,
    size: usize,
) {
    assert!(
        0 < shift && shift < size,
        "Expected: 0 < shift < size: found shift={}, size={}",
        shift,
        size,
    );
    assert!(
        dim < size,
        "Expected: dim < size: found dim={}, size={}",
        dim,
        size,
    );
}

/// Internal implementation of `roll_dim` that does not canonicalize dimensions or shifts.
///
/// ## Parameters
///
/// - `x`: The input tensor.
/// - `shift`: The number of positions to shift; must be (0 < shift < size).
/// - `dim`: The dimension to roll; must be a valid index for the tensor's shape.
///
/// ## Returns
///
/// A new tensor with the specified dimension rolled by the given shift amount.
#[inline(always)]
fn _unchecked_roll_dim<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    shift: usize,
    dim: usize,
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    let size = x.shape().dims[dim];

    #[cfg(debug_assertions)]
    _unchecked_roll_dim_contract(shift, dim, size);

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

    // This is a fair amount of complexity, which could be replaced
    // by a simple canonicalization of `dims` and wrapping of `shifts`.
    // The work is done here to ensure that any roll operation
    // which could be a no-op is a no-op; simplifying the accounting
    // needed by backend-specific implementations of the inner roll op.

    let item_count = dims.len();

    let shape = x.shape().dims;

    // Accumulate the effective shifts for each dimension.
    let mut shift_accum: Vec<isize> = vec![0; shape.len()];
    for i in 0..item_count {
        let dim = canonicalize_dim(dims[i], D, false);
        shift_accum[dim] += shifts[i];
    }

    // Do this after we've checked the validity of `dims` and `shifts`.
    if x.shape().num_elements() == 0 {
        // If the tensor is empty, return it as is.
        return x;
    }

    let sizes = x.shape().dims;

    // Wrap the accumulated shifts, and filter out empty dimensions.
    let mut _dims: Vec<usize> = Vec::with_capacity(item_count);
    let mut _shifts: Vec<usize> = Vec::with_capacity(item_count);
    for dim in 0..item_count {
        let shift = wrap_idx(shift_accum[dim], sizes[dim]);
        if shift != 0 {
            _shifts.push(shift);
            _dims.push(dim);
        }
    }

    // If no shifts are needed, return the original tensor.
    if _shifts.is_empty() {
        return x;
    }

    // At this point:
    // - the roll is non-trivial (i.e., at least one accumulated shift is non-zero),
    // - `dims` contains the effective dimensions to roll, in index order,
    // - `shifts` contains the effective usize shifts for each dimension.
    _unchecked_roll(x, &_shifts, &_dims)
}

/// Contract for the `_unchecked_roll` operation.
///
/// ## Parameters
///
/// - `shifts`: A slice of shifts corresponding to each dimension; must not be empty.
/// - `dims`: A slice of dimensions to roll; must be the same length as `shifts`, and must not contain repeats.
///
/// ## Panics
///
/// Panics if the shifts and dimensions do not align, or if dimensions contain repeats.
fn _unchecked_roll_contract(
    shifts: &[usize],
    dims: &[usize],
) {
    assert!(!shifts.is_empty());
    assert_eq!(
        shifts.len(),
        dims.len(),
        "Shifts and dimensions must align; found {} shifts and {} dims",
        shifts.len(),
        dims.len()
    );

    let mut _dims = dims.to_vec();
    _dims.dedup();

    assert_eq!(
        _dims.len(),
        dims.len(),
        "Dimensions must not contain repeats; found {} unique dims and {} total dims",
        _dims.len(),
        dims.len()
    )
}

/// `roll` internal implementation.
///
/// ## Parameters
///
/// - `x`: The input tensor.
/// - `shifts`: per-dimension shifts; must be non-empty,
///   and contain only non-zero values.
/// - `dims`: indices for `shifts`. Must be the same length as `shifts`,
///   must not contain repeats.
///
/// ## Returns
///
/// A new tensor with the specified dimensions rolled by the given shifts.
#[inline(always)]
fn _unchecked_roll<B: Backend, const D: usize, K>(
    x: Tensor<B, D, K>,
    shifts: &[usize],
    dims: &[usize],
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    #[cfg(debug_assertions)]
    _unchecked_roll_contract(shifts, dims);

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
