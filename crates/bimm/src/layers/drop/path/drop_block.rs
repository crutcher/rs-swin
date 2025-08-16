//! # `DropBlock` Layers

use crate::utility::expect_probability;
use burn::prelude::{Backend, Bool, Int, Tensor};
use burn::tensor::Distribution;
use burn::tensor::module::max_pool2d;
use std::cmp::min;

/// Configuration for `DropBlock`.
#[derive(Debug, Clone)]
pub struct DropBlockOptions {
    /// The drop probability.
    pub drop_prob: f32,

    /// The block size.
    pub block_size: usize,

    /// The gamma scale.
    pub gamma_scale: f32,

    /// Whether to use noise.
    pub with_noise: bool,

    /// Whether to drop batchwise.
    pub batchwise: bool,
}

impl Default for DropBlockOptions {
    fn default() -> Self {
        Self {
            drop_prob: 0.1,
            block_size: 7,
            gamma_scale: 1.0,
            with_noise: false,
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
        with_noise: bool,
    ) -> Self {
        Self { with_noise, ..self }
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

/// Clip Mask
#[allow(unused)]
fn clip_mask<B: Backend>(
    h: usize,
    w: usize,
    kernel: usize,
    device: &B::Device,
) -> Tensor<B, 2, Bool> {
    assert!(
        kernel <= h && kernel <= w,
        "kernel={kernel} must be <= h={h} and <= w={w}"
    );

    let start = kernel / 2;
    let end = (kernel - 1) / 2;

    let mask: Tensor<B, 2, Int> = Tensor::zeros([h, w], device);
    let mut mask = mask.bool();
    mask.inplace(|t| t.slice_fill([start..h - end, start..w - end], true));
    mask
}

#[inline(always)]
fn pool_block_seeds<B: Backend>(
    block_mask: Tensor<B, 4>,
    cbs: usize,
) -> Tensor<B, 4> {
    -max_pool2d(-block_mask, [cbs, cbs], [1, 1], [cbs / 2, cbs / 2], [1, 1])
}

/// `DropBlock`
///
/// See: See [DropBlock (Ghiasi, et all, 2018)](https://arxiv.org/pdf/1810.12890.pdf)
pub fn drop_block<B: Backend>(
    tensor: Tensor<B, 4>,
    options: &DropBlockOptions,
) -> Tensor<B, 4> {
    let [b, c, h, w] = tensor.shape().dims.to_vec().try_into().unwrap();
    let cbs = options.clipped_block_size(h, w);
    let device = &tensor.device();
    let dtype = tensor.dtype();

    let gamma_noise = Tensor::random(
        [if options.batchwise { 1 } else { b }, c, h, w],
        Distribution::Uniform(0.0, 1.0),
        device,
    )
    .add_scalar(1.0 - options.clipped_gamma(h, w))
    .cast(dtype);

    let seeds = clip_mask(h, w, cbs, device)
        .unsqueeze_dims::<4>(&[0, 1])
        .float()
        .lower(gamma_noise)
        .float()
        .cast(dtype);

    let block_mask = pool_block_seeds(seeds, cbs);

    let tensor = tensor * block_mask.clone();

    if options.with_noise {
        let normal_noise = Tensor::random(
            [if options.batchwise { 1 } else { b }, c, h, w],
            Distribution::Normal(0.0, 1.0),
            device,
        )
        .cast(dtype);

        tensor + normal_noise * (1.0 - block_mask)
    } else {
        let numel = block_mask.shape().num_elements() as f64;
        let normalize_scale = numel / block_mask.sum().add_scalar(1e-7).cast(dtype);

        tensor * normalize_scale.unsqueeze()
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
    fn test_drop_block_options() {
        let options = DropBlockOptions::default();
        assert_eq!(options.drop_prob, 0.1);
        assert_eq!(options.block_size, 7);
        assert_eq!(options.gamma_scale, 1.0);
        assert_eq!(options.with_noise, false);
        assert_eq!(options.batchwise, false);

        let options = options.with_drop_prob(0.2);
        assert_eq!(options.drop_prob, 0.2);

        let options = options.with_block_size(10);
        assert_eq!(options.block_size, 10);

        let options = options.with_gamma_scale(0.5);
        assert_eq!(options.gamma_scale, 0.5);

        let options = options.with_noise(true);
        assert_eq!(options.with_noise, true);

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

    #[test]
    fn test_clip_mask_odd() {
        type B = NdArray;
        let device = Default::default();

        let h = 7;
        let w = 9;
        let kernel = 3;

        let mask = clip_mask::<B>(h, w, kernel, &device);
        mask.to_data().assert_eq(
            &TensorData::from([
                [O, O, O, O, O, O, O, O, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, X, X, X, X, X, X, X, O],
                [O, O, O, O, O, O, O, O, O],
            ]),
            true,
        );
    }

    #[test]
    fn test_clip_mask_even() {
        type B = NdArray;
        let device = Default::default();

        let h = 7;
        let w = 9;
        let kernel = 2;

        let mask = clip_mask::<B>(h, w, kernel, &device);
        println!("mask: {:?}", mask);
        mask.to_data().assert_eq(
            &TensorData::from([
                [O, O, O, O, O, O, O, O, O],
                [O, X, X, X, X, X, X, X, X],
                [O, X, X, X, X, X, X, X, X],
                [O, X, X, X, X, X, X, X, X],
                [O, X, X, X, X, X, X, X, X],
                [O, X, X, X, X, X, X, X, X],
                [O, X, X, X, X, X, X, X, X],
            ]),
            true,
        );
    }
}
