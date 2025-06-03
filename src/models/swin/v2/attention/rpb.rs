use crate::models::swin::v2::attention::{
    window_attention_relative_position_index, window_log1p_relative_offset_grid,
};
use burn::config::Config;
use burn::module::Module;
use burn::nn;
use burn::prelude::{Backend, Int, Tensor};
// use burn::tensor::BasicOps;
use burn::tensor::activation::sigmoid;

/// Common introspection interface for relative position bias modules.
pub trait RelativePositionBiasMeta {
    fn base(&self) -> f64;
    fn num_heads(&self) -> usize;
    fn window_shape(&self) -> [usize; 2];

    fn window_height(&self) -> usize {
        self.window_shape()[0]
    }

    fn window_width(&self) -> usize {
        self.window_shape()[1]
    }
}

/// Configuration for the relative position bias module.
#[derive(Config, Debug, Copy)]
pub struct RelativePositionBiasConfig {
    #[config(default = 8.0)]
    pub base: f64,

    pub num_heads: usize,

    pub window_shape: [usize; 2],
}

impl RelativePositionBiasMeta for RelativePositionBiasConfig {
    fn base(&self) -> f64 {
        self.base
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn window_shape(&self) -> [usize; 2] {
        self.window_shape
    }
}

impl RelativePositionBiasConfig {
    /// Initializes an `OffsetGridRelativePositionBias` module.
    ///
    /// ## Arguments
    ///
    /// * `device`: The device on which the module will be created.
    ///
    /// ## Returns
    ///
    /// An `OffsetGridRelativePositionBias` module.
    #[inline(always)]
    #[must_use]
    pub fn init_offset_grid_rpb<B: Backend>(
        &self,
        device: &B::Device,
    ) -> OffsetGridRelativePositionBias<B> {
        OffsetGridRelativePositionBias {
            base: self.base,
            num_heads: self.num_heads,
            window_shape: self.window_shape,

            rel_coords_table: window_log1p_relative_offset_grid(
                self.window_shape,
                self.base,
                device,
            ),
            rel_index: window_attention_relative_position_index::<B>(self.window_shape, device),

            cbp: ContinuousPositionBiasMlpConfig::new(self.num_heads).init(device),
        }
    }
}

/// An offset grid relative position bias module.
///
/// Published/Used in SWIN-Transformer v2.
#[derive(Module, Debug)]
pub struct OffsetGridRelativePositionBias<B: Backend> {
    pub base: f64,
    pub num_heads: usize,
    pub window_shape: [usize; 2],

    pub rel_coords_table: Tensor<B, 3>,
    pub rel_index: Tensor<B, 2, Int>,

    pub cbp: ContinuousPositionBiasMlp<B>,
}

impl<B: Backend> RelativePositionBiasMeta for OffsetGridRelativePositionBias<B> {
    fn base(&self) -> f64 {
        self.base
    }
    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn window_shape(&self) -> [usize; 2] {
        self.window_shape
    }
}

impl<B: Backend> OffsetGridRelativePositionBias<B> {
    /// Returns the learned relative position bias.
    ///
    /// This is hashed such that all pairs of locations in the window with the same
    /// inter-pair relative offset will have the same bias across all heads.
    ///
    /// ## Returns
    ///
    /// A 3D tensor of shape (num_heads, height * width, height * width)
    /// containing the relative position bias for each head and position
    /// pair.
    #[must_use]
    pub fn forward(&self) -> Tensor<B, 3> {
        let [h, w] = self.window_shape;
        let hw = h * w;

        let lookup_table = self.cbp.forward(self.rel_coords_table.clone());
        // 2*h-1, 2*w-1, heads

        let rpb_table = lookup_table.reshape([-1, self.num_heads as i32]);
        // 2*((h-1) * (w-1)), heads

        let idx = self.rel_index.clone().reshape([-1]);
        // (h*w)^2

        let rpb = rpb_table.select(0, idx);
        // (h*w)^2, heads

        let rpb = rpb.reshape([hw, hw, self.num_heads]);
        let rpb = rpb.permute([2, 0, 1]);
        // heads, h*w, h*w

        sigmoid(rpb).mul_scalar(2.0 * self.base)
        // heads, h*w, h*w
    }
}

/// Common introspection interface for continuous position bias MLPs.
pub trait ContinuousPositionBiasMlpMeta {
    /// Returns the hidden dimension of the MLP.
    fn d_hidden(&self) -> usize;

    /// Returns the number of heads.
    fn num_heads(&self) -> usize;
}

#[derive(Config, Debug, Copy)]
pub struct ContinuousPositionBiasMlpConfig {
    /// The hidden dimension of the MLP.
    #[config(default = 512)]
    pub d_hidden: usize,

    /// The number of heads.
    pub num_heads: usize,
}

impl ContinuousPositionBiasMlpMeta for ContinuousPositionBiasMlpConfig {
    fn d_hidden(&self) -> usize {
        self.d_hidden
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }
}

/// A multi-layer perceptron (MLP) for continuous position bias.
#[derive(Module, Debug)]
pub struct ContinuousPositionBiasMlp<B: Backend> {
    l1: nn::Linear<B>,
    l2: nn::Linear<B>,
}

impl<B: Backend> ContinuousPositionBiasMlpMeta for ContinuousPositionBiasMlp<B> {
    fn d_hidden(&self) -> usize {
        self.l1.weight.dims()[1]
    }

    fn num_heads(&self) -> usize {
        self.l2.weight.dims()[1]
    }
}

impl ContinuousPositionBiasMlpConfig {
    /// Initializes the MLP with the given device.
    fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> ContinuousPositionBiasMlp<B> {
        ContinuousPositionBiasMlp {
            l1: nn::LinearConfig::new(2, self.d_hidden).init(device),

            l2: nn::LinearConfig::new(self.d_hidden, self.num_heads)
                .with_bias(false)
                .init(device),
        }
    }
}

impl<B: Backend> ContinuousPositionBiasMlp<B> {
    /// Applies the MLP to the input tensor.
    ///
    /// ## Arguments
    ///
    /// * `x`: A tensor of ``(..., 2)`` of the relative log-offset coordinates table.
    ///
    /// ## Returns
    ///
    /// A tensor of ``(..., num_heads)`` of the learned bias table.
    #[must_use]
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        let x = self.l1.forward(x);
        let x = burn::tensor::activation::relu(x);

        self.l2.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::models::swin::v2::attention::{
        window_attention_relative_position_index, window_log1p_relative_offset_grid,
    };
    use burn::backend::NdArray;
    use burn_contracts::assert_tensor;

    #[test]
    fn test_og_rpb() {
        let device = Default::default();

        let window_shape = [3, 2];
        let num_heads = 8;

        let config = RelativePositionBiasConfig::new(num_heads, window_shape);
        let rpb = config.init_offset_grid_rpb::<NdArray>(&device);

        assert_eq!(rpb.base(), 8.0);
        assert_eq!(rpb.num_heads(), num_heads);
        assert_eq!(rpb.window_shape(), window_shape);
        assert_eq!(rpb.window_height(), window_shape[0]);
        assert_eq!(rpb.window_width(), window_shape[1]);

        rpb.rel_coords_table.to_data().assert_eq(
            &window_log1p_relative_offset_grid::<NdArray>(window_shape, 8.0, &device).to_data(),
            true,
        );

        rpb.rel_index.to_data().assert_eq(
            &window_attention_relative_position_index::<NdArray>(window_shape, &device).to_data(),
            true,
        );

        let table = rpb.forward();
        // heads, h*w, h*w

        assert_tensor(&table)
            .unpacks_shape(
                [],
                "heads (h w) (h w)",
                &[
                    ("heads", num_heads),
                    ("h", window_shape[0]),
                    ("w", window_shape[1]),
                ],
            )
            .unwrap();
    }
}
