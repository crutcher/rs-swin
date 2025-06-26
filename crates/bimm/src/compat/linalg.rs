use burn::prelude::{Backend, Tensor};

/// Computes the L2 norm of a tensor along a specified dimension.
///
/// See: burn @"0.18.0" - "burn::linalg::l2_norm"
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the norm over.
///
/// # Returns
///
/// The L2 norm of the input tensor.
pub fn l2_norm<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: usize,
) -> Tensor<B, D> {
    x.abs().powi_scalar(2).sum_dim(dim).sqrt()
}

/// Computes the L2 normalization of a tensor along a specified dimension.
///
/// See: burn @"0.18.0" - "burn::linalg::vector_normalize"
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the normalization over.
/// * `eps` - A small value to avoid division by zero.
///
/// # Returns
///
/// The L2 normalized tensor.
#[cfg(not(feature = "burn_0_18_0"))]
pub fn l2_normalize<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: usize,
    eps: f64,
) -> Tensor<B, D> {
    x.clone() / l2_norm(x.clone(), dim).clamp_min(eps)
}

/// Computes the L2 normalization of a tensor along a specified dimension.
///
/// See: burn @"0.18.0" - "burn::linalg::vector_normalize"
///
/// # Arguments
///
/// * `x` - The input tensor.
/// * `dim` - The dimension to compute the normalization over.
/// * `eps` - A small value to avoid division by zero.
///
/// # Returns
///
/// The L2 normalized tensor.
#[cfg(feature = "burn_0_18_0")]
pub fn l2_normalize<B: Backend, const D: usize>(
    x: Tensor<B, D>,
    dim: usize,
    eps: f64,
) -> Tensor<B, D> {
    use burn::linalg::{Norm, vector_normalize};
    vector_normalize(x, Norm::L2, eps)
}
