use crate::compat::dims::canonicalize_dims;
use burn::prelude::{Backend, Tensor};
use burn::tensor::{BasicOps, Slice};

/// Roll operation.
///
/// ## Source
/// ```python
/// def roll(a: TensorLikeType, shifts: DimsType, dims: DimsType = ()) -> TensorLikeType:
///     """Reference implementation of :func:`torch.roll`."""
///     dims = utils.canonicalize_dims(a.ndim, dims)
///     # ATen specifies int[1] type for shifts and dims which expands integers to tuples of length 1
///     if not isinstance(shifts, Iterable):
///         shifts = (shifts,)
///     if not isinstance(dims, Iterable):
///         dims = (dims,)
///
///     # Avoid modulo by zero
///     if a.numel() == 0:
///         # Keeping this as ref for now as FakeTensor runs into some issues with complex tensors
///         return a.clone()
///
///     if a.dim() == 0 and len(dims) > 0:
///         raise IndexError(
///             f"Dimension specified as {dims[0]} but tensor has no dimensions"
///         )
///
///     len_shifts = len(shifts)
///     len_dims = len(dims)
///     if len_shifts != 1 or len_dims != 1:
///         if len_shifts == 0:
///             raise RuntimeError("`shifts` required")
///         # Takes care of the case when dims is not specified (default)
///         # By default, the tensor is flattened before shifting, after which the original shape is restored
///         if len_dims == 0 and len_shifts == 1:
///             return torch.roll(torch.flatten(a), shifts, 0).view(a.shape)
///         if len_shifts != len_dims:
///             raise RuntimeError(
///                 f"shifts and dimensions must align. shifts: {len_shifts}, dims: {len_dims}"
///             )
///         assert len_dims > 1
///         tail_shifts = shifts[1:]
///         tail_dims = dims[1:]
///         first_dim_rolled = torch.roll(a, (shifts[0],), dims[0])
///         return torch.roll(first_dim_rolled, tail_shifts, tail_dims)
///
///     # This path is taken when only one dimension is rolled
///     # For example to get `first_dim_rolled` above
///     dim = dims[0]
///     size = a.shape[dim]
///     start = (size - shifts[0]) % size
///     idx = torch.arange(size, device=a.device)
///     return a.index_select(dim, torch.fmod(start + idx, size))
/// ```
pub fn roll<B: Backend, const D: usize, K>(
    a: Tensor<B, D, K>,
    shifts: &[isize],
    dims: &[isize],
) -> Tensor<B, D, K>
where
    K: BasicOps<B>,
{
    let dims = canonicalize_dims(D, dims, false);

    // Avoid modulo by zero
    if a.shape().num_elements() == 0 {
        return a;
    }

    if a.shape().num_dims() == 0 && !dims.is_empty() {
        panic!(
            "Dimension specified as {} but tensor has no dimensions",
            dims[0]
        );
    }

    if shifts.len() > 1 {
        if dims.is_empty() && shifts.len() == 1 {
            let shape = a.shape();
            return roll(a.reshape([-1]), &[shifts[0]], &[0]).reshape(shape);
        }

        if dims.len() != shifts.len() {
            panic!(
                "shifts and dimensions must align. shifts: {}, dims: {}",
                shifts.len(),
                dims.len()
            );
        }

        let dims: Vec<isize> = dims.iter().map(|d| *d as isize).collect::<Vec<_>>();

        assert!(dims.len() > 1);
        let tail_shifts = &shifts[1..];
        let tail_dims = &dims[1..];
        let first_dim_rolled = roll(a, &[shifts[0]], &[dims[0]]);

        return roll(first_dim_rolled, tail_shifts, tail_dims);
    }

    let dim = dims[0];
    let size = a.shape().dims[0];

    let start = shifts[0];
    let start: usize = if start < 0 {
        (size as isize + start) as usize % size
    } else {
        (start as usize) % size
    };

    // If the start is 0, we can return the original tensor.
    if start == 0 {
        return a;
    }

    // a. Split the tensor into two chunks along the roll dimension;
    // b. Re-order the chunks.

    let mut r = Vec::with_capacity(D);
    for _i in 0..D {
        r.push(Slice::from(..));
    }

    let parts = a.clone().split_with_sizes(Tensor::from([start, (size - start) as usize], a.device()), dim);
    Tensor::cat(vec![parts[1].clone(), parts[0].clone()], dim)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::{Int, TensorData};
    use burn::tensor::Tensor;

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
            .assert_eq(&TensorData::from([[4, 5, 3], [1, 2, 0]]), false);
    }
}
