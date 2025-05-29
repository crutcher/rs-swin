use crate::compat::dims::{canonicalize_dim, wrap_idx};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;
use std::f64;

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
    use burn::prelude::{Int, TensorData};
    use burn::tensor::Tensor;

    #[test]
    fn test_float_arange() {
        let device = Default::default();
        let start: f64 = 3.0;
        let end: f64 = -1.0 - f64::EPSILON;

        let actual = float_arange::<NdArray>(start, end, None, &device);
        println!("{:?}", actual);

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
        println!("{:?}", actual);

        actual
            .to_data()
            .assert_eq(&TensorData::from([0.0, 0.25, 0.5, 0.75, 1.0]), false);
    }

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
