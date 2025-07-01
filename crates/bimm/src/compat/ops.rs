use crate::compat::indexing::{AsIndex, canonicalize_dim, wrap_idx};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;
use std::f64;

/// Roll operation along a specific dimension.
///
/// ## Parameters
///
/// - `tensor`: The input tensor.
/// - `shift`: The number of positions to shift; supports negative values and wraps around.
/// - `dim`: The dimension to roll; supports negative indexing.
///
/// ## Returns
///
/// A new tensor with the specified dimension rolled by the given shift amount.
#[must_use]
#[cfg(not(feature = "burn_0_18_0"))]
pub fn roll_dim<B: Backend, const D: usize, K, I>(
    tensor: Tensor<B, D, K>,
    shift: I,
    dim: I,
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
    I: AsIndex,
{
    let dim = canonicalize_dim(dim, D, false);
    let size = tensor.shape().dims[dim];
    if size == 0 {
        // If the dimension is empty, return the tensor as is.
        return tensor;
    }

    let shift = wrap_idx(shift, size);
    if shift == 0 {
        // If the shift is zero, return the tensor as is.
        return tensor;
    }

    _unchecked_roll_dim(tensor, shift, dim)
}

#[must_use]
#[cfg(feature = "burn_0_18_0")]
pub fn roll_dim<B: Backend, const D: usize, K, I>(
    x: Tensor<B, D, K>,
    shift: I,
    dim: I,
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
    I: AsIndex,
{
    x.roll_dim(shift, dim)
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
#[must_use]
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
    {
        assert!(
            0 < shift && shift < size,
            "Expected: 0 < shift < size: found shift={shift}, size={size}",
        );
        assert!(
            dim < x.shape().num_dims(),
            "Expected: dim < num_dims: found dim={dim}, num_dims={size}",
        );
    }
    let size = x.shape().dims[dim];

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
/// - `shifts`: A slice of shifts corresponding to each dimension;
///   supports negative values and wraps around.
/// - `dims`: A slice of dimensions to roll; supports negative indexing.
///
/// ## Returns
///
/// A new tensor with the specified dimensions rolled by the given shifts.
#[must_use]
#[cfg(not(feature = "burn_0_18_0"))]
pub fn roll<B: Backend, const D: usize, K, I>(
    tensor: Tensor<B, D, K>,
    shifts: &[I],
    dims: &[I],
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
    I: AsIndex,
{
    assert_eq!(
        dims.len(),
        shifts.len(),
        "Dimensions and shifts must align; found dims={dims:#?}, shifts={shifts:#?}",
    );

    // This is a fair amount of complexity, which could be replaced
    // by a simple canonicalization of `dims` and wrapping of `shifts`.
    // The work is done here to ensure that any roll operation
    // which could be a no-op is a no-op; simplifying the accounting
    // needed by backend-specific implementations of the inner roll op.

    let item_count = dims.len();

    let shape = tensor.shape().dims;

    // Accumulate the effective shifts for each dimension.
    let mut accumulated_shifts: Vec<isize> = vec![0; shape.len()];
    for i in 0..item_count {
        let dim = canonicalize_dim(dims[i], D, false);
        accumulated_shifts[dim] += shifts[i].index();
    }

    // Do this after we've checked the validity of `dims` and `shifts`.
    if tensor.shape().num_elements() == 0 {
        // If the tensor is empty, return it as is.
        return tensor;
    }

    // Wrap the accumulated shifts, and filter out empty dimensions.
    let mut effective_dims: Vec<usize> = Vec::with_capacity(item_count);
    let mut effective_shifts: Vec<usize> = Vec::with_capacity(item_count);
    for dim in 0..shape.len() {
        // `wrap_index` should inline, and has a fast-exit path for zero shifts.
        let shift = wrap_idx(accumulated_shifts[dim], shape[dim]);
        if shift == 0 {
            continue;
        }

        effective_dims.push(dim);
        effective_shifts.push(shift);
    }

    // If no shifts are needed, return the original tensor.
    if effective_shifts.is_empty() {
        return tensor;
    }

    // At this point:
    // - `dims` contains the effective dimensions to roll, in index order,
    // - `shifts` contains the effective usize shifts for each dimension.
    // - Every shift is non-zero, and less than the size of the corresponding dimension.
    _unchecked_roll(tensor, &effective_shifts, &effective_dims)
}
#[must_use]
#[cfg(feature = "burn_0_18_0")]
pub fn roll<B: Backend, const D: usize, K, I>(
    x: Tensor<B, D, K>,
    shifts: &[I],
    dims: &[I],
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
    I: AsIndex,
{
    x.roll(shifts, dims)
}

/// Contract for the `_unchecked_roll` operation.
///
/// ## Parameters
///
/// - `shifts`: A slice of shifts corresponding to each dimension; must not be empty.
/// - `dims`: A slice of dimensions to roll; must be the same length as `shifts`,
///   and must not contain repeats.
///
/// ## Panics
///
/// Panics if the shifts and dimensions do not align, or if dimensions contain repeats.
#[inline(always)]
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
/// - `tensor`: The input tensor.
/// - `shifts`: per-dimension shifts; must be non-empty,
///   and contain only non-zero values.
/// - `dims`: indices for `shifts`. Must be the same length as `shifts`,
///   must not contain repeats.
///
/// ## Returns
///
/// A new tensor with the specified dimensions rolled by the given shifts.
#[inline(always)]
#[must_use]
fn _unchecked_roll<B: Backend, const D: usize, K>(
    tensor: Tensor<B, D, K>,
    shifts: &[usize],
    dims: &[usize],
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    #[cfg(debug_assertions)]
    {
        assert!(!shifts.is_empty());
        assert_eq!(
            shifts.len(),
            dims.len(),
            "Shifts and dimensions must align; found {} shifts and {} dims",
            shifts.len(),
            dims.len()
        );

        let mut unique_dims = dims.to_vec();
        unique_dims.dedup();

        assert_eq!(
            unique_dims.len(),
            dims.len(),
            "Dimensions must not contain repeats; found {} unique dims and {} total dims",
            unique_dims.len(),
            dims.len()
        )
    }

    let tensor = _unchecked_roll_dim(tensor, shifts[0], dims[0]);

    if dims.len() == 1 {
        tensor
    } else {
        _unchecked_roll(tensor, &shifts[1..], &dims[1..])
    }
}

/// Create a vector with evenly spaced floating point values.
///
/// This function generates a vector starting from `start`, ending at `end`, and incrementing by `step`.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (exclusive).
/// - `step`: An optional step value. If not provided, defaults to `1.0` if `start < end`, or `-1.0` if `start > end`.
///
/// # Returns
///
/// A vector containing the generated floating point values.
#[must_use]
pub fn float_vec_arange(
    start: f64,
    end: f64,
    step: Option<f64>,
) -> Vec<f64> {
    assert_ne!(start, end);
    let step = if start < end {
        let step = step.unwrap_or(1.0);
        if step <= 0.0 {
            panic!("Step must be positive when start < end");
        }
        step
    } else {
        let step = step.unwrap_or(-1.0);
        if step >= 0.0 {
            panic!("Step must be negative when start > end");
        }
        step
    };

    let mut values: Vec<f64> = Vec::new();
    loop {
        let acc = start + values.len() as f64 * step;
        if (step > 0.0 && acc > end) || (step < 0.0 && acc < end) {
            break;
        }
        values.push(acc);
    }

    values
}

/// Create a vector with evenly spaced floating point values.
///
/// This function generates a vector with `num` values starting from `start`, ending at `end`, and evenly spaced.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (inclusive).
/// - `num`: The number of points to generate in the range.
///
/// # Returns
///
/// A vector containing the generated floating point values.
#[must_use]
pub fn float_vec_linspace(
    start: f64,
    end: f64,
    num: usize,
) -> Vec<f64> {
    assert!(num > 0, "Number of points must be positive");

    if num == 1 {
        return vec![start];
    }

    let step = (end - start) / (num as f64 - 1.0);

    let end = if step > 0.0 {
        end + f64::EPSILON // Avoid floating point precision issues
    } else {
        end - f64::EPSILON // Avoid floating point precision issues
    };

    float_vec_arange(start, end, Some(step))
}
/// Create a 1D tensor with evenly spaced floating point values.
///
/// This function generates a tensor with values starting from `start`, ending at `end`, and incrementing by `step`.
/// If `step` is not provided, it defaults to `1.0` if `start < end`, or `-1.0` if `start > end`.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (exclusive).
/// - `step`: An optional step value. If not provided, defaults to `1.0` or `-1.0` based on the order of `start` and `end`.
///
/// # Returns
///
/// A 1D tensor containing the generated floating point values.
#[must_use]
pub fn float_arange<B: Backend>(
    start: f64,
    end: f64,
    step: Option<f64>,
    device: &B::Device,
) -> Tensor<B, 1> {
    let values = float_vec_arange(start, end, step);
    Tensor::from_data(values.as_slice(), device)
}

/// Create a 1D tensor with evenly spaced floating point values.
///
/// This function generates a tensor with `num` values starting from `start`, ending at `end`, and evenly spaced.
///
/// # Parameters
///
/// - `start`: The starting value of the range.
/// - `end`: The end value of the range (inclusive).
/// - `num`: The number of points to generate in the range.
///
/// # Returns
///
/// A 1D tensor containing the generated floating point values.
#[must_use]
pub fn float_linspace<B: Backend>(
    start: f64,
    end: f64,
    num: usize,
    device: &B::Device,
) -> Tensor<B, 1> {
    let values = float_vec_linspace(start, end, num);
    Tensor::from_data(values.as_slice(), device)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::TensorData;
    use burn::tensor::Tensor;

    #[test]
    fn test_float_arange() {
        let device = Default::default();
        let start: f64 = 3.0;
        let end: f64 = -1.0 - f64::EPSILON;

        let actual = float_arange::<NdArray>(start, end, None, &device);
        println!("{actual:?}");

        actual
            .to_data()
            .assert_eq(&TensorData::from([3.0, 2.0, 1.0, 0.0, -1.0]), false);
    }

    #[test]
    fn test_float_linspace() {
        let device = Default::default();
        let start: f64 = 0.0;
        let end: f64 = 1.0;
        let num: usize = 5;

        let actual = float_linspace::<NdArray>(start, end, num, &device);
        println!("{actual:?}");

        actual
            .to_data()
            .assert_eq(&TensorData::from([0.0, 0.25, 0.5, 0.75, 1.0]), false);
    }

    #[test]
    fn test_roll() {
        let input = Tensor::<NdArray, 2>::from([[0, 1, 2], [3, 4, 5]]);

        // No-op shift:
        roll(input.clone(), &[0, 0], &[0, 1])
            .to_data()
            .assert_eq(&input.clone().to_data(), false);

        roll(input.clone(), &[1, -1], &[0, 1])
            .to_data()
            .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);

        roll(input.clone(), &[-1, 1], &[1, 0])
            .to_data()
            .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);

        roll(input.clone(), &[2 * 32 + 1, 3 * (-400) - 1], &[0, 1])
            .to_data()
            .assert_eq(&TensorData::from([[5, 3, 4], [2, 0, 1]]), false);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_too_big() {
        let input = Tensor::<NdArray, 2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = roll(input.clone(), &[1], &[-3]);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_too_small() {
        let input = Tensor::<NdArray, 2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = roll(input.clone(), &[1], &[2]);
    }

    #[should_panic]
    #[test]
    fn test_roll_shift_size_mismatch() {
        let input = Tensor::<NdArray, 2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll with a shift size that doesn't match the number of dimensions should panic
        let _d = roll(input.clone(), &[1, 2], &[0]);
    }

    #[test]
    fn test_roll_dim() {
        let input = Tensor::<NdArray, 2>::from([[0, 1, 2], [3, 4, 5]]);

        roll_dim(input.clone(), 1, 0)
            .to_data()
            .assert_eq(&TensorData::from([[3, 4, 5], [0, 1, 2]]), false);

        roll_dim(input.clone(), -1, 1)
            .to_data()
            .assert_eq(&TensorData::from([[2, 0, 1], [5, 3, 4]]), false);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_dim_too_big() {
        let input = Tensor::<NdArray, 2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = roll_dim(input.clone(), 1, 2);
    }

    #[should_panic]
    #[test]
    fn test_roll_dim_dim_too_small() {
        let input = Tensor::<NdArray, 2>::from([[0, 1, 2], [3, 4, 5]]);

        // Attempting to roll on a dimension that doesn't exist should panic
        let _d = roll_dim(input.clone(), 1, -3);
    }
}
