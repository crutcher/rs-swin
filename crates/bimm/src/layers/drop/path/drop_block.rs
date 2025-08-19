//! # `DropBlock` Layers

use crate::utility::burn::shape::shape_to_ranges;
use crate::utility::probability::expect_probability;
use crate::utility::zspace::expect_point_bounds_check;
use bimm_contracts::unpack_shape_contract;
use burn::prelude::{Backend, Shape, Tensor};
use burn::tensor::module::max_pool2d;
use burn::tensor::{DType, Distribution};

/// Clip Range.
#[derive(Debug, Clone)]
pub struct ClampConfig {
    /// The minimum value.
    min: Option<f64>,

    /// The maximum value.
    max: Option<f64>,
}

impl ClampConfig {
    /// Apply the clamp.
    pub fn clamp<B: Backend, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match (self.min, self.max) {
            (Some(min), Some(max)) => tensor.clamp(min, max),
            (Some(min), None) => tensor.clamp_min(min),
            (None, Some(max)) => tensor.clamp_max(max),
            (None, None) => tensor,
        }
    }
}

/// Noise Configuration.
#[derive(Debug, Clone)]
pub struct NoiseConfig {
    /// The noise distribution.
    distribution: Distribution,

    /// The noise clip range.
    clamp: Option<ClampConfig>,
}

impl NoiseConfig {
    /// Generate noise.
    pub fn noise<B: Backend, const D: usize, S>(
        &self,
        shape: S,
        device: &B::Device,
    ) -> Tensor<B, D>
    where
        S: Into<Shape>,
    {
        let noise = Tensor::random(shape.into(), self.distribution, device);
        match &self.clamp {
            None => noise,
            Some(clamp_cfg) => clamp_cfg.clamp(noise),
        }
    }
}

/// Configuration for `DropBlock`.
#[derive(Debug, Clone)]
pub struct DropBlockOptions {
    /// The drop probability.
    pub drop_prob: f64,

    /// The block size.
    pub kernel: [usize; 2],

    /// The gamma scale.
    pub gamma_scale: f64,

    /// The noise configuration.
    pub noise_cfg: Option<NoiseConfig>,

    /// Whether to drop batchwise.
    pub batchwise: bool,

    /// Permit partial blocks at the edges, faster.
    pub messy: bool,
}

impl Default for DropBlockOptions {
    fn default() -> Self {
        Self {
            drop_prob: 0.1,
            kernel: [7; 2],
            gamma_scale: 1.0,
            noise_cfg: None,
            batchwise: false,
            messy: false,
        }
    }
}

impl DropBlockOptions {
    /// Set the drop probability.
    pub fn with_drop_prob(
        self,
        drop_prob: f64,
    ) -> Self {
        Self {
            drop_prob: expect_probability(drop_prob),
            ..self
        }
    }

    /// Set the block size.
    pub fn with_block_size(
        self,
        block_size: usize,
    ) -> Self {
        self.with_kernel([block_size; 2])
    }

    /// Sets the kernel size.
    pub fn with_kernel(
        self,
        kernel: [usize; 2],
    ) -> Self {
        Self { kernel, ..self }
    }

    /// Clip the kernel to fit within the shape.
    pub fn clipped_kernel(
        &self,
        shape: [usize; 2],
    ) -> [usize; 2] {
        let [h, w] = shape;
        let [kh, kw] = self.kernel;
        [std::cmp::min(h, kh), std::cmp::min(w, kw)]
    }

    /// Set the gamma scale.
    pub fn with_gamma_scale(
        self,
        gamma_scale: f64,
    ) -> Self {
        Self {
            gamma_scale,
            ..self
        }
    }

    /// Set whether to use noise.
    pub fn with_noise(
        self,
        noise: Option<NoiseConfig>,
    ) -> Self {
        Self {
            noise_cfg: noise,
            ..self
        }
    }

    /// Set whether to drop batchwise.
    pub fn with_batchwise(
        self,
        batchwise: bool,
    ) -> Self {
        Self { batchwise, ..self }
    }

    /// Compute the adjusted gamma rate.
    ///
    /// Gamma is the adjusted probability that any given point is the midpoint
    /// of a dropped block; given the desired `drop_rate`, the block size, and the input size.
    ///
    /// ## Arguments
    ///
    /// - `shape`: the shape of the target tensor.
    #[inline]
    pub fn gamma(
        &self,
        shape: [usize; 2],
    ) -> f64 {
        let [h, w] = shape;
        let [kh, kw] = self.clipped_kernel(shape);

        (self.gamma_scale * self.drop_prob * ((h * w) as f64))
            / ((kh * kw) as f64)
            / (((h - kh + 1) * (w - kw + 1)) as f64)
    }

    /// Compute the gamma noise for the given noise batch shape.
    ///
    /// # Args
    ///
    /// - `shape`: the target noise batch shape.
    ///
    /// # Returns
    ///
    /// Gamma noise, sampled at ``self.gamma([h, w])`` rate.
    pub fn gamma_noise<B: Backend>(
        &self,
        noise_shape: [usize; 4],
        device: &B::Device,
    ) -> Tensor<B, 4> {
        let [_, _, h, w] = noise_shape;
        let gamma = self.gamma([h, w]);
        Tensor::random(noise_shape, Distribution::Bernoulli(gamma), device)
    }
}

/// Build a filter of kernel midpoints.
///
/// `1.0` at midpoints of kernels of size ``(h, w)``;
/// `0.0` everywhere else.
///
/// This predicts the kernel midpoints that ``conv2d`` (and related kernel functions)
/// would place a kernel.
///
/// The *midpoint* of a kernel is computed as ``size / 2``:
/// * the midpoint of odd kernels is the middle: `mid(3) == 1`
/// * the midpoint of even kernels is the first point in the second half: `mid(4) == 2`
///
/// # Argument
///
/// - `shape`: the mask shape.
/// - `kernel`: the shape of the kernel.
/// - `device`: the device to construct the mask on.
#[inline]
pub fn conv2d_kernel_midpoint_filter<B: Backend>(
    shape: [usize; 2],
    kernel: [usize; 2],
    device: &B::Device,
) -> Tensor<B, 2> {
    expect_point_bounds_check(&kernel, &[0; 2], &shape);
    let region = [
        (kernel[0] / 2)..shape[0] - ((kernel[0] - 1) / 2),
        (kernel[1] / 2)..shape[1] - ((kernel[1] - 1) / 2),
    ];
    Tensor::zeros(shape, device).slice_fill(region, true)
}

/// Convert drop block gamma noise to a selection filter.
///
/// This is a deterministic internal component of `drop_block`.
///
/// # Arguments
///
/// * `selected_blocks` - Input selection noise;
///   `1.0` at the midpoints of selected blocks to drop,
///   `0.0` everywhere else. Expected to be gamma noise.
/// * `kernel_shape` - the shape of the kernel.
/// * `messy` - permit partial blocks at the edges, faster.
pub fn drop_block_2d_drop_filter_<B: Backend>(
    selected_blocks: Tensor<B, 4>,
    kernel_shape: [usize; 2],
    messy: bool,
) -> Tensor<B, 4> {
    let [_, _, h, w] = unpack_shape_contract!(["b", "c", "h", "w"], &selected_blocks);
    let [kh, kw] = kernel_shape;

    assert!(
        kh <= h && kw <= w,
        "Kernel size ({kh}, {kw}) is larger than input size ({h}, {w})",
    );

    let dtype = selected_blocks.dtype();
    let device = &selected_blocks.device();

    let mut selection = selected_blocks;

    if !messy {
        selection = selection
            * conv2d_kernel_midpoint_filter([h, w], kernel_shape, device)
                .unsqueeze_dims::<4>(&[0, 1])
                .cast(dtype);
    }

    selection = max_pool2d(selection, kernel_shape, [1, 1], [kh / 2, kw / 2], [1, 1]);

    // Clip even-kernel padding artifacts.
    if (kh % 2) == 0 || (kw % 2) == 0 {
        let mut ranges = shape_to_ranges::<4>(selection.shape());
        ranges[2].start = ((kh % 2) == 0) as usize;
        ranges[3].start = ((kw % 2) == 0) as usize;

        selection = selection.slice(ranges);
    }

    selection
}

/// `drop_block_2d`
///
/// Drops block kernels from a tensor; many options.
///
/// Dropped values can be resampled from a noise distribution,
/// kept values can be re-normalized.
/// The drop probability, block size, and several performance/quality tradeoffs
/// can be configured.
///
/// Based upon [DropBlock (Ghiasi, et all, 2018)](https://arxiv.org/pdf/1810.12890.pdf);
/// inspired also by the `python-image-models` implementation.
///
/// # Arguments
///
/// * `tensor` - the tensor to operate on.
/// * `options` - the algorithm options.
pub fn drop_block_2d<B: Backend>(
    tensor: Tensor<B, 4>,
    options: &DropBlockOptions,
) -> Tensor<B, 4> {
    let [b, c, h, w] = tensor.shape().dims();
    let kernel = options.clipped_kernel([h, w]);

    let t_shape = tensor.shape();
    let device = &tensor.device();
    let dtype = tensor.dtype();

    let noise_shape = [if options.batchwise { 1 } else { b }, c, h, w];

    let gamma_noise = options.gamma_noise(noise_shape, device);

    let drop_filter: Tensor<B, 4> =
        drop_block_2d_drop_filter_(gamma_noise, kernel, options.messy).cast(dtype);
    let keep_filter: Tensor<B, 4> = 1.0 - drop_filter.clone();

    if let Some(noise_cfg) = &options.noise_cfg {
        // Fill in the dropped regions with sampled noise.
        let noise: Tensor<B, 4> = noise_cfg.noise(noise_shape, device).cast(dtype);
        let noise = noise * drop_filter;

        tensor * keep_filter.expand(t_shape.clone()) + noise.expand(t_shape)
    } else {
        // Rescale the kept regions to normalize to 1.0.
        let count = keep_filter.shape().num_elements() as f32;
        let total = keep_filter.clone().cast(DType::F32).sum();
        let norm_scale = count / total.add_scalar(1e-7);

        tensor * keep_filter.expand(t_shape.clone()) * norm_scale.cast(dtype).expand(t_shape)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::NdArray;
    use burn::prelude::TensorData;

    #[test]
    fn test_drop_block_2d() {
        type B = NdArray;
        let device = Default::default();

        let shape = [1, 1, 7, 9];

        let tensor: Tensor<B, 4> = Tensor::ones(shape, &device);

        let drop = drop_block_2d(
            tensor,
            &DropBlockOptions::default()
                .with_noise(Some(NoiseConfig {
                    distribution: Distribution::Default,
                    clamp: None,
                }))
                .with_drop_prob(0.1)
                .with_block_size(2),
        );
        println!("drop\n{:?}", drop);
    }

    #[test]
    fn test_seed_origin_mask() {
        let shape = [7, 9];
        let kernel_shape = [2, 3];

        type B = NdArray;
        let device = Default::default();

        let mask: Tensor<B, 2> = conv2d_kernel_midpoint_filter(shape, kernel_shape, &device);
        mask.to_data().assert_eq(
            &TensorData::from([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            ]),
            false,
        );
    }

    #[test]
    fn test_drop_block_options() {
        let options = DropBlockOptions::default();
        assert_eq!(options.drop_prob, 0.1);
        assert_eq!(options.kernel, [7; 2]);
        assert_eq!(options.gamma_scale, 1.0);
        assert!(options.noise_cfg.is_none());
        assert_eq!(options.batchwise, false);

        let options = options.with_drop_prob(0.2);
        assert_eq!(options.drop_prob, 0.2);

        let options = options.with_block_size(10);
        assert_eq!(options.kernel, [10; 2]);

        let options = options.with_gamma_scale(0.5);
        assert_eq!(options.gamma_scale, 0.5);

        let options = options.with_batchwise(true);
        assert_eq!(options.batchwise, true);
    }

    #[test]
    fn test_gamma() {
        let options = DropBlockOptions::default()
            .with_drop_prob(0.1)
            .with_gamma_scale(1.2)
            .with_kernel([2, 3]);

        let shape = [7, 9];
        let [h, w] = shape;
        let [kh, kw] = options.kernel;

        let total_size = (h * w) as f64;

        let gamma = options.gamma([h, w]);

        let expected = (options.gamma_scale * options.drop_prob * total_size)
            / ((kh * kw) as f64)
            / (((h - kh + 1) * (w - kw + 1)) as f64);

        assert_eq!(gamma, expected);
    }
}
