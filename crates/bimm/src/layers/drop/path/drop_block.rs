//! # `DropBlock` Layers

use crate::layers::drop::path::zspace::expect_point_bounds_check;
use crate::utility::probability::expect_probability;
use burn::prelude::{Backend, Bool, Int, Shape, Tensor};
use burn::tensor::module::max_pool2d;
use burn::tensor::{Distribution, RangesArg};
use std::cmp::{min};
use std::ops::Range;

/// Clip Range.
#[derive(Debug, Clone)]
pub struct ClampConfig {
    /// The minimum value.
    min: Option<f32>,

    /// The maximum value.
    max: Option<f32>,
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
    where S: Into<Shape>,
    {
        let noise = Tensor::random(
            shape.into(),
            self.distribution,
            device,
        );
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
    pub drop_prob: f32,

    /// The block size.
    pub block_size: usize,

    /// The gamma scale.
    pub gamma_scale: f32,

    /// The noise configuration.
    pub noise_cfg: Option<NoiseConfig>,

    /// Whether to drop batchwise.
    pub batchwise: bool,
}

impl Default for DropBlockOptions {
    fn default() -> Self {
        Self {
            drop_prob: 0.1,
            block_size: 7,
            gamma_scale: 1.0,
            noise_cfg: None,
            batchwise: false,
        }
    }
}

impl DropBlockOptions {
    /// Set the drop probability.
    pub fn with_drop_prob(
        self,
        drop_prob: f32,
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
        Self { block_size, ..self }
    }

    /// Set the gamma scale.
    pub fn with_gamma_scale(
        self,
        gamma_scale: f32,
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

    /// The clipped block size.
    ///
    /// ## Arguments
    ///
    /// - `h`: the height.
    /// - `w`: the width.
    #[inline]
    pub fn clipped_block_size(
        &self,
        h: usize,
        w: usize,
    ) -> usize {
        min(self.block_size, min(h, w))
    }

    /// Compute the clipped gamma value for a ``(h, w)`` pair.
    ///
    /// ## Arguments
    ///
    /// - `h`: the height.
    /// - `w`: the width.
    #[inline]
    pub fn clipped_gamma(
        &self,
        h: usize,
        w: usize,
    ) -> f32 {
        let total_size = (h * w) as f32;
        let clipped_block_size = self.clipped_block_size(h, w) as f32;
        (self.gamma_scale * self.drop_prob * total_size)
            / clipped_block_size.powi(2)
            / (((w - self.block_size + 1) * (h * self.block_size + 1)) as f32)
    }
}

/// `DropBlock`
///
/// See: See [DropBlock (Ghiasi, et all, 2018)](https://arxiv.org/pdf/1810.12890.pdf)
pub fn drop_block_selection<B: Backend>(
    tensor: Tensor<B, 4>,
    selection: Tensor<B, 4>,
    options: &DropBlockOptions,
) -> Tensor<B, 4> {
    let dtype = tensor.dtype();
    let device = &tensor.device();

    let [b, c, h, w] = tensor.shape().dims.to_vec().try_into().unwrap();

    // TODO: gamma is being miscalculated / I don't understand the expected usage.
    // Double check the source paper for the expected gamma value.
    //
    // Since drop_prob is meant to mean the probability of dropping (keeping?)
    // a pixel, but pixels are dropped by blocks, and blocks overlap,
    // gamma needs to be a function of the drop_prob, the number of blocks,
    // and the block size.
    //
    // Should gamma be the keep-a-seed prob, or the drop-a-seed prob?

    let block_bias = drop_filter_2d(selection, [options.block_size; 2], false);
    println!("block_bias:\n{:?}", block_bias);

    let tensor = tensor * block_bias.clone();

    if let Some(noise_cfg) = &options.noise_cfg {
        let normal_noise = noise_cfg.noise(
            [if options.batchwise { 1 } else { b }, c, h, w],
            device,
        ).cast(dtype);

        tensor + normal_noise * (1.0 - block_bias)
    } else {
        // TODO: scale normalization config?
        let numel = block_bias.shape().num_elements() as f64;
        let normalize_scale = numel / block_bias.sum().add_scalar(1e-7).cast(dtype);

        tensor * normalize_scale.unsqueeze()
    }
}

/// `DropBlock`
///
/// See: See [DropBlock (Ghiasi, et all, 2018)](https://arxiv.org/pdf/1810.12890.pdf)
pub fn drop_block<B: Backend>(
    tensor: Tensor<B, 4>,
    options: &DropBlockOptions,
) -> Tensor<B, 4> {
    let [b, c, h, w] = tensor.shape().dims.to_vec().try_into().unwrap();
    let device = &tensor.device();
    let dtype = tensor.dtype();

    let gamma = options.drop_prob;

    let noise: Tensor<B, 4> = Tensor::random(
        [if options.batchwise { 1 } else { b }, c, h, w],
        Distribution::Bernoulli(gamma as f64),
        device,
    )
    .cast(dtype);

    let tensor = drop_block_selection(tensor, noise, options);

    tensor

    /*
    if options.with_noise {
        let normal_noise = Tensor::random(
            [if options.batchwise { 1 } else { b }, c, h, w],
            Distribution::Normal(0.0, 1.0),
            device,
        )
        .cast(dtype);

        tensor + normal_noise * (1.0 - block_bias)
    } else {
        let numel = block_bias.shape().num_elements() as f64;
        let normalize_scale = numel / block_bias.sum().add_scalar(1e-7).cast(dtype);

        tensor * normalize_scale.unsqueeze()
    }
     */
}


/// Sub-Region Mask.
///
/// Constructs a `shape` based `Bool` mask, and marks the selected region `true`.
///
/// # Arguments
///
/// - `shape`: the shape of the mask.
/// - `region`: the subregion to mark; must be entirely within the `shape`.
/// - `device`: the device to construct the mask on.
///
/// # Returns
///
/// A new `Tensor<B, D, Bool>` mask.
#[inline]
pub fn sub_region_mask<B: Backend, S, const D: usize, R: RangesArg<D>>(
    shape: S,
    region: R,
    device: &B::Device,
) -> Tensor<B, D, Bool>
where
    S: Into<Shape>,
{
    let shape = shape.into();
    assert_eq!(
        shape.num_dims(),
        D,
        "Shape dims ({}) != region dims ({})",
        shape.num_dims(),
        D
    );

    // TODO: Construct directly as Bool when supported upstream.
    let mut mask = Tensor::<B, D, Int>::zeros(shape, device).bool();
    mask.inplace(|t| t.slice_fill(region, true));
    mask
}

/// Construct a kernel midpoint mask for regions of shape `kernel_shape`.
///
/// # Argument
///
/// - `shape`: the mask shape.
/// - `kernel_shape`: the shape of the kernel.
/// ` `device`: the device to construct the mask on.
#[inline]
pub fn kernel_midpoint_mask<B: Backend, S1, S2, const D: usize>(
    shape: S1,
    kernel_shape: S2,
    device: &B::Device,
) -> Tensor<B, D, Bool>
where
    B: Backend,
    S1: Into<Shape>,
    S2: Into<Shape>,
{
    let shape = shape.into();
    let kernel_shape = kernel_shape.into();
    expect_point_bounds_check(&kernel_shape.dims, &[0; D], &shape.dims);

    let region: [Range<usize>; D] = (0..D)
        .into_iter()
        .map(|i| {
            let start = kernel_shape.dims[i] / 2;
            let end = (kernel_shape.dims[i] - 1) / 2;
            start..shape.dims[i] - end
        })
        .collect::<Vec<_>>()
        .try_into()
        .unwrap();

    sub_region_mask(shape, region, device)
}

/// Returns a tensor with values (0,0, 1.0):
/// * 0.0 where a pixel should be dropped, and
/// * 1.0 where a pixel should be kept.
pub fn drop_filter_2d<B: Backend>(
    selection: Tensor<B, 4>,
    kernel_shape: [usize; 2],
    inverse: bool,
) -> Tensor<B, 4> {
    let mask: Tensor<B, 2, Bool> = kernel_midpoint_mask(
        &selection.shape().dims[2..],
        kernel_shape.clone(),
        &selection.device(),
    );
    let mask: Tensor<B, 4, Bool> = mask.unsqueeze_dims::<4>(&[0, 1]);

    // keep seeds with prob gamma.
    let drop_seeds = mask.float() * selection;

    let [h, w] = kernel_shape;

    // TODO: Native bool mask convolution would add speed here.
    let mut drop_coverage = max_pool2d(
        drop_seeds,
        kernel_shape.clone(),
        [1, 1],
        [h / 2, w / 2],
        [1, 1],
    );

    // Clip even-kernel padding artifacts.
    if (h % 2) == 0 || (w % 2) == 0 {
        let mut ranges: [Range<usize>; 4] = drop_coverage
            .shape()
            .dims
            .iter()
            .map(|s| 0..*s)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();
        ranges[2].start = ((h % 2) == 0) as usize;
        ranges[3].start = ((w % 2) == 0) as usize;

        drop_coverage = drop_coverage.slice(ranges);
    }


    if inverse {
        drop_coverage
    } else {
        drop_coverage.neg().add_scalar(1)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::NdArray;
    use burn::prelude::TensorData;
    use hamcrest::prelude::*;

    const X: bool = true;
    const O: bool = false;

    #[test]
    fn test_sub_region_mask() {
        let shape = [4, 5];

        type B = NdArray;
        let device = Default::default();

        let mask: Tensor<B, 2, Bool> = sub_region_mask(shape, [.., ..], &device);
        mask.to_data().assert_eq(
            &TensorData::from([
                [X, X, X, X, X],
                [X, X, X, X, X],
                [X, X, X, X, X],
                [X, X, X, X, X],
            ]),
            true,
        );

        let mask: Tensor<B, 2, Bool> = sub_region_mask(shape, [1.., 2..], &device);
        mask.to_data().assert_eq(
            &TensorData::from([
                [O, O, O, O, O],
                [O, O, X, X, X],
                [O, O, X, X, X],
                [O, O, X, X, X],
            ]),
            true,
        );

        let mask: Tensor<B, 2, Bool> = sub_region_mask(shape, [1..3, 2..4], &device);
        mask.to_data().assert_eq(
            &TensorData::from([
                [O, O, O, O, O],
                [O, O, X, X, O],
                [O, O, X, X, O],
                [O, O, O, O, O],
            ]),
            true,
        );
    }

    #[test]
    fn test_seed_origin_mask() {
        let shape = [7, 9];
        let kernel_shape = [2, 3];

        type B = NdArray;
        let device = Default::default();

        let mask: Tensor<B, 2, Bool> = kernel_midpoint_mask(shape, kernel_shape, &device);
        mask.to_data().assert_eq(
            &TensorData::from([
                [O, O, O, O, O, O, O, O, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
            ]),
            true,
        );
    }

    #[test]
    fn test_drop_block() {
        type B = NdArray;
        let device = Default::default();

        let input: Tensor<B, 4> = Tensor::<B, 2>::from_data(
            [
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
                [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
            ],
            &device,
        )
        .unsqueeze_dims::<4>(&[0, 1]);

        let selection: Tensor<B, 4> = Tensor::<B, 2>::from_data(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            &device,
        )
        .unsqueeze_dims::<4>(&[0, 1]);

        let drop = drop_block_selection(
            input,
            selection,
            &DropBlockOptions::default()
                .with_noise(Some(NoiseConfig {
                    distribution: Distribution::Normal(0.0, 1.0),
                    clamp: Some(ClampConfig {
                        min: Some(0.1),
                        max: Some(0.7),
                    }),
                }))
                .with_block_size(2)
                .with_drop_prob(0.1),
        );

        println!("drop:\n{:?}", drop.squeeze_dims::<2>(&[0, 1]));
    }

    #[test]
    fn test_drop_block_options() {
        let options = DropBlockOptions::default();
        assert_eq!(options.drop_prob, 0.1);
        assert_eq!(options.block_size, 7);
        assert_eq!(options.gamma_scale, 1.0);
        assert!(options.noise_cfg.is_none());
        assert_eq!(options.batchwise, false);

        let options = options.with_drop_prob(0.2);
        assert_eq!(options.drop_prob, 0.2);

        let options = options.with_block_size(10);
        assert_eq!(options.block_size, 10);

        let options = options.with_gamma_scale(0.5);
        assert_eq!(options.gamma_scale, 0.5);

        let options = options.with_batchwise(true);
        assert_eq!(options.batchwise, true);
    }

    #[test]
    fn test_clipped_block_size() {
        let options = DropBlockOptions::default().with_block_size(7);

        assert_eq!(options.clipped_block_size(10, 12), 7);
        assert_eq!(options.clipped_block_size(3, 10), 3);
        assert_eq!(options.clipped_block_size(10, 3), 3);
    }

    #[test]
    fn test_clipped_gamma() {
        let options = DropBlockOptions::default()
            .with_drop_prob(0.1)
            .with_gamma_scale(1.2)
            .with_block_size(7);

        let h = 10;
        let w = 12;
        let total_size = (h * w) as f32;

        let cbs = options.clipped_block_size(h, w) as f32;

        let expected = (options.gamma_scale * options.drop_prob * total_size)
            / cbs.powi(2)
            / (((w - options.block_size + 1) * (h * options.block_size + 1)) as f32);

        assert_that!(options.clipped_gamma(10, 12), close_to(expected, 1e-5));
    }
}
