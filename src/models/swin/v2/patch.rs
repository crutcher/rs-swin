use crate::models::swin::v2::windowing::{window_partition, window_reverse};
use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Tensor};
use burn_contracts::assert_tensor;

/// Metadata for PatchMerging.
pub trait PatchMergingMeta {
    /// Input feature dimension size.
    fn d_input(&self) -> usize;

    /// Output feature dimension size.
    fn d_output(&self) -> usize {
        2 * self.d_input()
    }

    /// Input resolution (height, width).
    fn input_resolution(&self) -> [usize; 2];

    /// Input height.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// Input width.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// Output resolution (height, width).
    fn output_resolution(&self) -> [usize; 2] {
        let [h, w] = self.input_resolution();
        [h / 2, w / 2]
    }

    /// Output height.
    fn output_height(&self) -> usize {
        self.input_height() / 2
    }

    /// Output width.
    fn output_width(&self) -> usize {
        self.input_width() / 2
    }
}

/// Configuration for PatchMerging.
#[derive(Config, Debug)]
pub struct PatchMergingConfig {
    /// Input resolution (height, width).
    input_resolution: [usize; 2],

    /// Input feature dimension size.
    d_input: usize,
}

impl PatchMergingMeta for PatchMergingConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }
}

impl PatchMergingConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PatchMerging<B> {
        let [h, w] = self.input_resolution;
        assert!(
            h % 2 == 0 && w % 2 == 0,
            "Input resolution must be divisible by 2: {:?}",
            self.input_resolution
        );

        PatchMerging {
            input_resolution: self.input_resolution,
            reduction: LinearConfig::new(2 * self.d_output(), self.d_output())
                .with_bias(false)
                .init(device),
            norm: LayerNormConfig::new(self.d_output()).init(device),
        }
    }
}

/// SWIN-Transformer v2 PatchMerging module.
///
/// This module accepts ``(B, (H * W), C)`` inputs, and then:
/// - Collates interleaved patches of size ``(H/2, W/2)`` into ``(B, H/2 * W/2, 4 * C)``.
/// - Applies a linear layer to reduce the feature dimension to ``2 * C``.
/// - Applies layer normalization.
/// - Yields output of shape ``(B, H/2 * W/2, 2 * C)``.
///
/// See: https://github.com/microsoft/Swin-Transformer/blob/main/models/swin_transformer_v2.py
#[derive(Module, Debug)]
pub struct PatchMerging<B: Backend> {
    input_resolution: [usize; 2],
    reduction: Linear<B>,
    norm: LayerNorm<B>,
}

impl<B: Backend> PatchMergingMeta for PatchMerging<B> {
    fn d_input(&self) -> usize {
        self.reduction.weight.dims()[0] / 4
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }
}

impl<B: Backend> PatchMerging<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [b, h, w] = assert_tensor(&x)
            .unpacks_shape(
                ["b", "h", "w"],
                "b (h w) d_in",
                &[("h", self.input_height()), ("w", self.input_width())],
            )
            .unwrap();

        let h2 = h / 2;
        let w2 = w / 2;
        let h2w2 = h2 * w2;

        let x = collate_patches(x, h, w);

        let x = self.reduction.forward(x);

        let x = self.norm.forward(x);

        #[cfg(debug_assertions)]
        assert_tensor(&x).has_named_dims([
            ("B", b),
            ("(H/2)*(W/2)", h2w2),
            ("d_out", self.d_output()),
        ]);

        x
    }
}

/// Collate patches from a tensor of shape ``(B, (H * W), C)`` to ``(B, H/2 * W/2, 4 * C)``.
///
/// This splits the input into 4 interleaved patches, each of size ``(H/2, W/2)``;
/// and then concatenates them along the last dimension.
///
/// # Arguments
///
/// * `x` - Input tensor of shape (B, (H * W), C).
/// * `h` - Height of the input tensor.
/// * `w` - Width of the input tensor.
///
/// # Returns
///
/// * A tensor of shape (B, H/2 * W/2, 4 * C).
pub fn collate_patches<B: Backend>(
    x: Tensor<B, 3>,
    h: usize,
    w: usize,
) -> Tensor<B, 3> {
    let [b, h, w, c] = assert_tensor(&x)
        .unpacks_shape(["b", "h", "w", "c"], "b (h w) c", &[("h", h), ("w", w)])
        .unwrap();

    let h2 = h / 2;
    let w2 = w / 2;
    let h2w2 = h2 * w2;

    // TODO(crutcher): re-using `window_partition` requires us to double-reshape.
    // Is it worth writing a trait to permit windowing on 3D and 4D tensors?
    // In SWIN Source; window_partition is *always* immediately resized.
    let x = x.reshape([b, h, w, c]);
    let x = window_partition(x, 2);

    x.reshape([b, h2w2, 4 * c])
}

/// Decollate patches from a tensor of shape (B, H/2 * W/2, 4 * C) to (B, (H * W), C).
///
/// This splits the input into 4 patches, each of size (H/2, W/2);
/// and interleaves them along the height and width dimensions.
///
/// The inverse operation of `collate_patches`.
///
/// # Arguments
///
/// * `x` - Input tensor of shape (B, H/2 * W/2, 4 * C).
/// * `h` - Height of the input tensor.
/// * `w` - Width of the input tensor.
///
/// # Returns
///
/// * A tensor of shape (B, (H * W), C).
pub fn decollate_patches<B: Backend>(
    x: Tensor<B, 3>,
    h: usize,
    w: usize,
) -> Tensor<B, 3> {
    let h2 = h / 2;
    let w2 = w / 2;

    let [b, c] = assert_tensor(&x)
        .unpacks_shape(["b", "c"], "b (h2 w2) c", &[("h2", h2), ("w2", w2)])
        .unwrap();

    let c = c / 4;

    let x = x.reshape([b * h2 * w2, 2, 2, c]);
    let x = window_reverse(x, 2, h, w);

    x.reshape([b, h * w, c])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Backend;
    use burn::tensor::Distribution;

    #[test]
    fn test_collate_patches() {
        let b = 2;
        let h = 4;
        let w = 6;
        let c = 5;

        let device = Default::default();

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::<NdArray, 3>::random([b, h * w, c], distribution, &device);

        let y = collate_patches(x.clone(), h, w);
        assert_tensor(&y).has_named_dims([
            ("B", b),
            ("(H/2)*(W/2)", (h / 2) * (w / 2)),
            ("4*d_in", 4 * c),
        ]);

        decollate_patches(y.clone(), h, w)
            .into_data()
            .assert_eq(&x.into_data(), true);
    }

    #[test]
    fn test_patch_merging() {
        impl_test_patch_merging::<NdArray>();
    }

    fn impl_test_patch_merging<B: Backend>() {
        let device: B::Device = Default::default();

        let b = 2;
        let h = 12;
        let w = 8;
        let c = 3;

        let config = PatchMergingConfig {
            input_resolution: [h, w],
            d_input: c,
        };
        let patch_merging = config.init::<B>(&device);

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::random([b, h * w, c], distribution, &device);

        let y = patch_merging.forward(x.clone());
        assert_tensor(&y).has_named_dims([
            ("B", b),
            ("H/2*W/2", (h / 2) * (w / 2)),
            ("4*C", 2 * c),
        ]);
    }
}
