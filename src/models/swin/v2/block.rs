use crate::compat::ops::roll;
use crate::layers::drop::{DropPath, DropPathConfig};
use crate::models::swin::v2::attention::swmsa::sw_attn_mask;
use crate::models::swin::v2::attention::{
    WindowAttention, WindowAttentionConfig, WindowAttentionMeta,
};
use crate::models::swin::v2::mlp::{BlockMlp, BlockMlpConfig, BlockMlpMeta};
use crate::models::swin::v2::windowing::{window_partition, window_reverse};
use burn::config::Config;
use burn::module::Module;
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;

/// Applies an inner function under conditional cyclic shift.
///
/// This is used for shifted window attention. When `swa_enabled` is true,
/// it cyclically shifts the input tensor by `shift_size` in the last two dimensions,
/// applies the function `f`, and then reverses the cyclic shift.
///
/// When `swa_enabled` is false, it simply applies the function `f` without any shift.
///
/// ## Parameters
///
/// * `x` - Input tensor of shape (B, H, W, C).
/// * `f` - Function to apply on the shifted tensor.
///
/// ## Returns
///
/// A new tensor of the same shape as `x`, with the function `f` applied after cyclic shifting.
#[must_use]
#[inline(always)]
fn with_shift<B: Backend, F, K>(
    x: Tensor<B, 4, K>,
    shift: isize,
    f: F,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
    F: FnOnce(Tensor<B, 4, K>) -> Tensor<B, 4, K>,
{
    let dims = [1, 2];

    // Cyclic shift for shifted window attention.
    let x = if shift != 0 {
        roll(x, &[-shift, -shift], &dims)
    } else {
        x
    };

    let x = f(x);

    // Reverse cyclic shift.
    if shift != 0 {
        roll(x, &[shift, shift], &dims)
    } else {
        x
    }
}

/// Common introspection interface for TransformerBlock.
pub trait SwinTransformerBlockMeta {
    /// Get the input dimension size.
    fn d_input(&self) -> usize;

    /// Get the input resolution.
    fn input_resolution(&self) -> [usize; 2];

    /// Get the input height.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// Get the input width.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// Get the output dimension size.
    fn d_output(&self) -> usize {
        self.d_input()
    }

    /// Get the output resolution.
    fn output_resolution(&self) -> [usize; 2] {
        self.input_resolution()
    }

    /// Get the output height.
    fn output_height(&self) -> usize {
        self.input_height()
    }

    /// Get the output width.
    fn output_width(&self) -> usize {
        self.input_width()
    }

    /// Get the number of attention heads.
    fn num_heads(&self) -> usize;

    /// Window size for window attention.
    fn window_size(&self) -> usize;

    /// Shift size for shifted window attention; 0 means no shift.
    fn shift_size(&self) -> usize;

    /// Is shifted window attention enabled?
    fn swa_enabled(&self) -> bool {
        self.shift_size() > 0
    }

    /// Whether to enable QKV bias.
    fn enable_qkv_bias(&self) -> bool;

    /// Dropout rate for MLP.
    fn drop_rate(&self) -> f64;

    /// Dropout rate for attention.
    fn attn_drop_rate(&self) -> f64;

    /// Ratio of hidden dimension to input dimension in MLP.
    fn mlp_ratio(&self) -> f64;

    /// Drop path rate for stochastic depth.
    fn drop_path_rate(&self) -> f64;
}

/// Configuration for TransformerBlock.
#[derive(Config, Debug)]
pub struct SwinTransformerBlockConfig {
    pub d_input: usize,

    pub input_resolution: [usize; 2],

    pub num_heads: usize,

    #[config(default = 7)]
    pub window_size: usize,

    #[config(default = 0)]
    pub shift_size: usize,

    #[config(default = 4.0)]
    pub mlp_ratio: f64,

    #[config(default = true)]
    pub enable_qkv_bias: bool,

    #[config(default = 0.0)]
    pub drop_rate: f64,

    #[config(default = 0.0)]
    pub attn_drop_rate: f64,

    #[config(default = 0.0)]
    pub drop_path_rate: f64,
    // TODO: act_layer, norm_layer
}

impl SwinTransformerBlockMeta for SwinTransformerBlockConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn shift_size(&self) -> usize {
        self.shift_size
    }

    fn enable_qkv_bias(&self) -> bool {
        self.enable_qkv_bias
    }

    fn drop_rate(&self) -> f64 {
        self.drop_rate
    }

    fn attn_drop_rate(&self) -> f64 {
        self.attn_drop_rate
    }

    fn mlp_ratio(&self) -> f64 {
        self.mlp_ratio
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path_rate
    }
}

impl SwinTransformerBlockConfig {
    /// Initializes a new `SwinTransformerBlock`.
    ///
    /// # Parameters
    ///
    /// * `device` - The device on which the block will be created.
    ///
    /// # Returns
    ///
    /// A new `SwinTransformerBlock` instance configured with the specified parameters.
    #[must_use]
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> SwinTransformerBlock<B> {
        let [h, w] = self.input_resolution;
        assert!(
            h % self.window_size == 0 && w % self.window_size == 0,
            "Input resolution {:?} must be divisible by window size: {:?}",
            self.input_resolution,
            self.window_size
        );

        let hidden_dim = (self.d_input as f64 * self.mlp_ratio) as usize;
        let block_mlp = BlockMlpConfig::new(self.d_input)
            .with_d_hidden(Some(hidden_dim))
            .with_drop(self.drop_rate)
            .init(device);

        let win_attn = WindowAttentionConfig::new(
            self.d_input,
            [self.window_size, self.window_size],
            self.num_heads,
        )
        .with_enable_qkv_bias(self.enable_qkv_bias)
        .with_attn_drop(self.attn_drop_rate)
        .with_proj_drop(self.drop_rate)
        .init(device);

        let shift_mask = if self.shift_size == 0 {
            None
        } else {
            Some(
                sw_attn_mask(
                    self.input_resolution,
                    self.window_size,
                    self.shift_size,
                    device,
                )
                .float()
                .mul_scalar(-100.0),
            )
        };

        SwinTransformerBlock {
            input_resolution: self.input_resolution,
            window_size: self.window_size,
            shift_size: self.shift_size,
            shift_mask,
            drop_path: DropPathConfig::new()
                .with_drop_prob(self.drop_path_rate)
                .init(),
            norm1: LayerNormConfig::new(self.d_input).init(device),
            norm2: LayerNormConfig::new(self.d_input).init(device),
            win_attn,
            block_mlp,
        }
    }
}

/// Basic Swin Transformer Block.
///
/// Equivalent to the ``SwinTransformerBlock`` in the python source.
///
/// Applies one layer of Swin Transformer block with window attention and MLP.
#[derive(Module, Debug)]
pub struct SwinTransformerBlock<B: Backend> {
    pub input_resolution: [usize; 2],
    pub window_size: usize,

    pub shift_size: usize,
    pub shift_mask: Option<Tensor<B, 3>>,

    pub drop_path: DropPath,

    pub norm1: LayerNorm<B>,
    pub norm2: LayerNorm<B>,
    pub win_attn: WindowAttention<B>,
    pub block_mlp: BlockMlp<B>,
}

impl<B: Backend> SwinTransformerBlockMeta for SwinTransformerBlock<B> {
    fn d_input(&self) -> usize {
        self.win_attn.d_input()
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn num_heads(&self) -> usize {
        self.win_attn.num_heads()
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn shift_size(&self) -> usize {
        self.shift_size
    }

    fn enable_qkv_bias(&self) -> bool {
        self.win_attn.enable_qkv_bias()
    }

    fn drop_rate(&self) -> f64 {
        self.block_mlp.drop()
    }

    fn attn_drop_rate(&self) -> f64 {
        self.win_attn.attn_drop()
    }

    fn mlp_ratio(&self) -> f64 {
        self.block_mlp.d_hidden() as f64 / self.d_input() as f64
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path.drop_prob
    }
}

impl<B: Backend> SwinTransformerBlock<B> {
    /// Applies the forward pass on the input tensor.
    ///
    /// H and W are the height and width of the input resolution, and D is the input dimension.
    ///
    /// # Parameters
    ///
    /// * `x` - Input tensor of shape (B, H * W, D), where B is the batch size,
    ///
    /// # Returns
    ///
    /// A new tensor of shape (B, H * W, D) with the transformer block applied.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [h, w] = self.input_resolution;
        let [b, l, c] = x.dims();
        assert_eq!(
            l,
            h * w,
            "Expected input shape (B, H ({}) * W ({}), D), but got {:?}",
            h,
            w,
            x.dims()
        );

        let x = self.with_skip(x, |x| {
            let x = x.reshape([b, h, w, c]);

            let x = with_shift(x, self.shift_size as isize, |x| self.apply_window(x, c));
            // b, h, w, c

            let x = x.reshape([b, h * w, c]);
            self.norm1.forward(x)
        });
        // b, h * w, c

        let x = self.with_skip(x, |x| self.norm2.forward(self.block_mlp.forward(x)));

        assert_eq!(
            l,
            h * w,
            "Expected input shape (B, H ({}) * W ({}), D), but got {:?}",
            h,
            w,
            x.dims()
        );
        x
    }

    /// Applies an inner function under conditional stochastic residual/depth-skip connection.
    #[must_use]
    #[inline(always)]
    fn with_skip<const D: usize, F>(
        &self,
        x: Tensor<B, D>,
        f: F,
    ) -> Tensor<B, D>
    where
        F: FnOnce(Tensor<B, D>) -> Tensor<B, D>,
    {
        self.drop_path.with_skip(x, f)
    }

    /// Applies window attention to the input tensor.
    ///
    /// This function partitions the input tensor into windows, applies window attention,
    /// and then merges the windows back to the original shape.
    ///
    /// ## Parameters
    ///
    /// * `x` - Input tensor of shape (B, H, W, C).
    /// * `c` - Number of channels in the input tensor.
    ///
    /// ## Returns
    ///
    /// A new tensor of shape (B, H, W, C) with window attention applied.
    #[must_use]
    #[inline(always)]
    fn apply_window(
        &self,
        x: Tensor<B, 4>,
        c: usize,
    ) -> Tensor<B, 4> {
        let [h, w] = self.input_resolution;
        let ws = self.window_size as i32;
        let c = c as i32;

        // Partition into windows.
        let x_windows = window_partition(x, self.window_size);
        // b*nW, ws, ws, c
        let x_windows = x_windows.reshape([-1, ws * ws, c]);
        // b*nW, ws*ws, c

        let attn_windows = self.win_attn.forward(x_windows, self.shift_mask.clone());
        // b*nW, ws*ws, c

        // Merge windows back to the original shape.
        let attn_windows = attn_windows.reshape([-1, ws, ws, c]);
        window_reverse(attn_windows, self.window_size, h, w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_with_shift() {
        let device = Default::default();
        let b = 1;
        let h = 4;
        let w = 4;
        let c = 3;

        let distribution = burn::tensor::Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 4>::random([b, h, w, c], distribution, &device);

        let idx: Tensor<NdArray, 4> =
            Tensor::arange(0..input.shape().num_elements() as i64, &device)
                .reshape([b, h, w, c])
                .float();

        // No-op shift:
        with_shift(input.clone(), 0, |x| x + idx.clone())
            .to_data()
            .assert_eq(&(input.clone() + idx.clone()).to_data(), true);

        with_shift(input.clone(), 1, |x| x + idx.clone())
            .to_data()
            .assert_eq(
                &({
                    let x = input.clone();
                    let x = roll(x, &[-1, -1], &[1, 2]);
                    let x = x + idx.clone();
                    roll(x, &[1, 1], &[1, 2])
                })
                .to_data(),
                true,
            );
    }

    #[test]
    fn test_config() {
        let d_input = 128;
        let num_heads = 4;
        let input_resolution = [32, 32];

        let config = SwinTransformerBlockConfig::new(d_input, input_resolution, num_heads);

        assert_eq!(config.d_input, d_input);
        assert_eq!(config.input_resolution, input_resolution);
        assert_eq!(config.num_heads, num_heads);
        assert_eq!(config.window_size, 7);
        assert_eq!(config.shift_size, 0);
        assert_eq!(config.mlp_ratio, 4.0);
        assert!(config.enable_qkv_bias);
        assert_eq!(config.drop_path_rate, 0.0);
    }

    #[test]
    fn test_block() {
        let b = 1;
        let num_heads = 4;
        let channels_per_head = 3;
        let d_input = num_heads * channels_per_head;
        let window_size = 4;

        let h = 2 * window_size;
        let w = 3 * window_size;
        let input_resolution = [h, w];

        let config = SwinTransformerBlockConfig::new(d_input, input_resolution, num_heads)
            .with_window_size(window_size);

        let device = Default::default();

        let block = config.init::<NdArray>(&device);

        let distribution = burn::tensor::Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 3>::random([b, h * w, d_input], distribution, &device);

        let _output = block.forward(input.clone());

        assert_eq!(input.dims(), _output.dims());
    }
}
