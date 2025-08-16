//! # `DropBlock` Layers

use crate::utility::expect_probability;
use burn::prelude::{Backend, Bool, Int, Tensor};
use burn::tensor::Distribution;
use burn::tensor::grid::{GridIndexing, meshgrid};
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

/// The valid block map.
#[inline(always)]
fn valid_block_map<B: Backend>(
    block_size: usize,
    h: usize,
    w: usize,
    device: &B::Device,
) -> Tensor<B, 4, Bool> {
    let cbs = min(block_size, min(h, w));

    // FIXME: this reverses the `timm` implementation; (H, W) vs (W, H); find the bug.
    let [h_i, w_i]: [Tensor<B, 2, Int>; 2] = meshgrid(
        &[
            Tensor::arange(0..(h as i64), device),
            Tensor::arange(0..(w as i64), device),
        ],
        GridIndexing::Matrix,
    );
    let wa = w_i.clone().greater_equal_elem((cbs / 2) as i64);
    let wb = w_i.clone().lower_elem((w - ((cbs - 1) / 2)) as i64);
    let ha = h_i.clone().greater_equal_elem((cbs / 2) as i64);
    let hb = h_i.clone().lower_elem((h - ((cbs - 1) / 2)) as i64);

    let m = wa.bool_and(wb)
        .bool_and(ha)
        .bool_and(hb)
        .reshape([1, 1, h, w]);

    println!("vbm: {m:?}");

    m
}

#[inline(always)]
fn fuzz_block_mask<B: Backend>(
    block_mask: Tensor<B, 4>,
    gamma: f32,
    noise: Tensor<B, 4>,
) -> Tensor<B, 4, Bool> {
    (2.0 - gamma - block_mask + noise).greater_equal_elem(1.0)
}

#[inline(always)]
fn pool_block_mask<B: Backend>(
    block_mask: Tensor<B, 4>,
    cbs: usize,
) -> Tensor<B, 4> {
    -max_pool2d(
        -block_mask,
        [cbs, cbs],
        [1, 1],
        [cbs / 2, cbs / 2],
        [1, 1],
    )
}

fn drop_block_mask<B: Backend>(
    noise: Tensor<B, 4>,
    options: &DropBlockOptions,
) -> Tensor<B, 4> {
    let [h, w] = noise.shape().dims[2..4].try_into().unwrap();
    let device = &noise.device();
    let dtype = noise.dtype();

    let cbs = options.clipped_block_size(h, w);
    let gamma = options.clipped_gamma(h, w);

    // Restrict blocks to the feature map.
    let block_mask = valid_block_map(options.block_size, h, w, device)
        .float()
        .cast(dtype);
    let block_mask = fuzz_block_mask(block_mask, gamma, noise)
        .float()
        .cast(dtype);
    let block_mask = pool_block_mask(block_mask, cbs);

    block_mask
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

    let make_noise = || -> Tensor<B, 4> {
        Tensor::random(
            [if options.batchwise { 1 } else { b }, c, h, w],
            Distribution::Uniform(0.0, 1.0),
            device,
        )
            .cast(dtype)
    };

    let block_mask = drop_block_mask(make_noise(), options);

    let scale: Tensor<B, 4> = if options.with_noise {
        make_noise().cast(tensor.dtype()) * (1.0 - block_mask.clone())
    } else {
        ((block_mask.shape().num_elements() as f64) / block_mask.clone().sum().add_scalar(1e-7))
            .unsqueeze()
    };

    tensor * block_mask * scale
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
    fn test_valid_block_map() {
        type B = NdArray;
        let device = Default::default();

        let h = 7;
        let w = 9;
        let c = 1;

        let block_map: Tensor<B, 4, Bool> = valid_block_map(
            5, // block size
            7, // height
            9, // width
            &device,
        );

        block_map.clone().squeeze_dims::<2>(&[0, 1]).to_data().assert_eq(
            &TensorData::from([
                [O, O, O, O, O, O, O, O, O],
                [O, O, O, O, O, O, O, O, O],
                [O, O, X, X, X, X, X, O, O],
                [O, O, X, X, X, X, X, O, O],
                [O, O, X, X, X, X, X, O, O],
                [O, O, O, O, O, O, O, O, O],
                [O, O, O, O, O, O, O, O, O],
            ]),
            true,
        );

        let noise: Tensor<B, 4> = Tensor::random(
            [1, c, h, w],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let cbs = 5;
        let fbm = fuzz_block_mask(block_map.clone().float(), 1.0, noise);
        println!("fbm: {:?}", fbm);

        let pbm = pool_block_mask(block_map.clone().float(), cbs);
        println!("pbm: {:?}", pbm);
    }

    #[test]
    fn test_indices() {
        type B = NdArray;
        let device = Default::default();

        let h = 7;
        let w = 10;

        let [w_i, h_i]: [Tensor<B, 2, Int>; 2] = meshgrid(
            &[
                Tensor::arange(0..(w as i64), &device),
                Tensor::arange(0..(h as i64), &device),
            ],
            GridIndexing::Matrix,
        );

        println!("w_i: {:?}", w_i);
        println!("h_i: {:?}", h_i);
    }

    #[test]
    fn test_drop_block_mask() {
        type B = NdArray;
        let device = Default::default();

        let options = DropBlockOptions::default()
            .with_block_size(5);

        let h = 7;
        let w = 9;
        let c = 1;

        let noise: Tensor<B, 4> = Tensor::random(
            [1, c, h, w],
            Distribution::Uniform(0.0, 1.0),
            &device,
        );

        let mask = drop_block_mask(noise, &options);
        println!("mask: {:?}", mask);
        mask.clone().squeeze_dims::<2>(&[0, 1]).to_data().assert_eq(
            &TensorData::from([
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ]),
            false,
        );
    }
}
