use crate::layers::drop::{DropPath, DropPathConfig};
use crate::models::swin::v2::attention::swmsa::sw_attn_mask;
use crate::models::swin::v2::attention::{
    WindowAttention, WindowAttentionConfig, WindowAttentionMeta,
};
use crate::models::swin::v2::mlp::{Mlp, MlpConfig, MlpMeta};
use crate::models::swin::v2::windowing::{window_partition, window_reverse};
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, Tensor};

/// Common introspection interface for TransformerBlock.
pub trait TransformerBlockMeta {
    /// Get the input dimension size.
    fn d_input(&self) -> usize;

    /// Get the output dimension size.
    fn d_output(&self) -> usize {
        self.d_input()
    }

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
pub struct TransformerBlockConfig {
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

impl TransformerBlockMeta for TransformerBlockConfig {
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

impl TransformerBlockConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> TransformerBlock<B> {
        let [h, w] = self.input_resolution;
        assert!(
            h % self.window_size == 0 && w % self.window_size == 0,
            "Input resolution must be divisible by window size: {:?}",
            self.input_resolution
        );

        let attn_mask = if self.swa_enabled() {
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
        } else {
            None
        };

        let mlp_hidden_dim = (self.d_input as f64 * self.mlp_ratio) as usize;

        TransformerBlock {
            input_resolution: self.input_resolution,
            window_size: self.window_size,
            shift_size: self.shift_size,
            attn: WindowAttentionConfig::new(
                self.d_input,
                [self.window_size, self.window_size],
                self.num_heads,
            )
            .with_enable_qkv_bias(self.enable_qkv_bias)
            .with_attn_drop(self.attn_drop_rate)
            .with_proj_drop(self.drop_rate)
            .init(device),
            drop_path: DropPathConfig::new()
                .with_drop_prob(self.drop_path_rate)
                .init(),
            mlp: MlpConfig::new(self.d_input)
                .with_d_hidden(Some(mlp_hidden_dim))
                .with_drop(self.drop_rate)
                .init(device),
            attn_mask,
        }
    }
}

#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    pub input_resolution: [usize; 2],
    pub window_size: usize,
    pub shift_size: usize,

    // norm1
    // norm2
    pub attn: WindowAttention<B>,
    pub drop_path: DropPath,
    pub mlp: Mlp<B>,

    // nw, ws, ws, 1
    // nw, ws * ws
    // nw, 1, ws * ws
    // nw, 1, 1, ws * ws
    pub attn_mask: Option<Tensor<B, 3>>,
}

impl<B: Backend> TransformerBlockMeta for TransformerBlock<B> {
    fn d_input(&self) -> usize {
        self.attn.d_input()
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn num_heads(&self) -> usize {
        self.attn.num_heads()
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn shift_size(&self) -> usize {
        self.shift_size
    }

    fn enable_qkv_bias(&self) -> bool {
        self.attn.enable_qkv_bias()
    }

    fn drop_rate(&self) -> f64 {
        self.mlp.drop()
    }

    fn attn_drop_rate(&self) -> f64 {
        self.attn.attn_drop()
    }

    fn mlp_ratio(&self) -> f64 {
        self.mlp.d_hidden() as f64 / self.d_input() as f64
    }

    fn drop_path_rate(&self) -> f64 {
        self.drop_path.drop_prob
    }
}

impl<B: Backend> TransformerBlock<B> {
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let [h, w] = self.input_resolution;
        let [b, l, c] = x.dims();
        assert_eq!(l, h * w);

        let shortcut = x.clone();
        let x = x.reshape([b, h, w, c]);

        let x = if self.swa_enabled() {
            panic!("tensor.roll() is not implemented yet");
        } else {
            x
        };

        let x_windows = window_partition(x, self.window_size);
        let x_windows =
            x_windows.reshape([-1, (self.window_size * self.window_size) as i32, c as i32]);

        let attn_windows = self.attn.forward(x_windows, self.attn_mask.clone());

        let attn_windows = attn_windows.reshape([
            -1,
            self.window_size as i32,
            self.window_size as i32,
            c as i32,
        ]);
        let shifted_x = window_reverse(attn_windows, self.window_size, h, w);

        let x = if self.swa_enabled() {
            panic!("tensor.roll() is not implemented yet");
        } else {
            shifted_x
        };

        let x = x.reshape([b, h * w, c]);

        // TODO
        // let x = self.norm1.forward(x);

        let x = self.drop_path.forward(x);
        let x = shortcut + x;

        // TODO
        let x = self.mlp.forward(x);
        // let x = self.norm2.forward(x);

        self.drop_path.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;

    #[test]
    fn test_config() {
        let d_input = 128;
        let num_heads = 4;
        let input_resolution = [32, 32];

        let config = TransformerBlockConfig::new(d_input, input_resolution, num_heads);

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

        let config = TransformerBlockConfig::new(d_input, input_resolution, num_heads)
            .with_window_size(window_size);

        let device = Default::default();

        let block = config.init::<NdArray>(&device);

        let distribution = burn::tensor::Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 3>::random([b, h * w, d_input], distribution, &device);

        let _output = block.forward(input.clone());

        assert_eq!(input.dims(), _output.dims());
    }
}
