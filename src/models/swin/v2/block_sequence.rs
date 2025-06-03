use crate::models::swin::v2::swin_block::{
    ShiftedWindowTransformerBlock, ShiftedWindowTransformerBlockConfig,
    ShiftedWindowTransformerBlockMeta,
};
use burn::config::Config;
use burn::module::Module;
use burn::prelude::{Backend, Tensor};

/// Common introspection train for BasicLayer.
pub trait StochasticDepthTransformerBlockSequenceMeta {
    /// Returns the number of input channels for the layer.`
    fn d_input(&self) -> usize;

    /// Returns the image resolution of the input to the layer.
    fn input_resolution(&self) -> [usize; 2];

    /// Returns the height of the input to the layer.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// Returns the width of the input to the layer.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// Number of internal blocks in the layer.
    fn depth(&self) -> usize;

    /// Returns the number of heads in the layer.
    fn num_heads(&self) -> usize;

    /// Window size for window attention.
    fn window_size(&self) -> usize;

    /// Ratio of hidden dimension to input dimension in MLP.
    fn mlp_ratio(&self) -> f64;

    /// Whether to enable QKV bias.
    fn enable_qkv_bias(&self) -> bool;

    /// Dropout rate for MLP.
    fn drop_rate(&self) -> f64;

    /// Dropout rate for attention.
    fn attn_drop_rate(&self) -> f64;

    /// Returns the per-depth drop path rates.
    fn drop_path_rates(&self) -> Vec<f64>;

    // NOTE: downsample lifted to above this module.

    // TODO: norm_layer config, use_checkpoint, pretrained_window_size.
}

/// Config for BasicLayer.
#[derive(Config, Debug)]
pub struct StochasticDepthTransformerBlockSequenceConfig {
    /// Number of input channels.
    pub d_input: usize,

    /// Image resolution of the input.
    pub input_resolution: [usize; 2],

    /// Number of internal blocks in the layer.
    pub depth: usize,

    /// Number of heads in the layer.
    pub num_heads: usize,

    /// Window size for window attention.
    pub window_size: usize,

    /// Ratio of hidden dimension to input dimension in MLP.
    #[config(default = 4.0)]
    pub mlp_ratio: f64,

    /// Whether to enable QKV bias.
    #[config(default = true)]
    pub enable_qkv_bias: bool,

    /// Dropout rate for MLP.
    #[config(default = 0.0)]
    pub drop_rate: f64,

    /// Dropout rate for attention.
    #[config(default = 0.0)]
    pub attn_drop_rate: f64,

    #[config(default = 0.0)]
    pub common_drop_path_rate: f64,

    #[config(default = "None")]
    pub drop_path_rates: Option<Vec<f64>>,
}

impl StochasticDepthTransformerBlockSequenceMeta for StochasticDepthTransformerBlockSequenceConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn depth(&self) -> usize {
        self.depth
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn mlp_ratio(&self) -> f64 {
        self.mlp_ratio
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

    fn drop_path_rates(&self) -> Vec<f64> {
        if let Some(ref rates) = self.drop_path_rates {
            assert_eq!(rates.len(), self.depth);
            rates.clone()
        } else {
            // If no specific drop path rates are provided, use a common rate
            vec![self.common_drop_path_rate; self.depth]
        }
    }
}

impl StochasticDepthTransformerBlockSequenceConfig {
    /// Creates a common block configuration for the sequence.
    #[must_use]
    pub fn common_block_config(&self) -> ShiftedWindowTransformerBlockConfig {
        ShiftedWindowTransformerBlockConfig::new(
            self.d_input,
            self.input_resolution,
            self.num_heads,
        )
        .with_window_size(self.window_size)
        .with_mlp_ratio(self.mlp_ratio)
        .with_enable_qkv_bias(self.enable_qkv_bias)
        .with_drop_rate(self.drop_rate)
        .with_attn_drop_rate(self.attn_drop_rate)
    }

    /// Returns the block configs for each child.
    #[must_use]
    pub fn block_configs(&self) -> Vec<ShiftedWindowTransformerBlockConfig> {
        let drop_path_rates = self.drop_path_rates();
        assert_eq!(drop_path_rates.len(), self.depth);

        let shift = self.window_size / 2;

        let common_config = self.common_block_config();

        (0..self.depth)
            .map(|i| {
                common_config
                    .clone()
                    .with_shift_size(
                        // SW-SWA step?
                        if i % 2 == 0 { 0 } else { shift },
                    )
                    .with_drop_path_rate(drop_path_rates[i])
            })
            .collect()
    }

    /// Creates a new `BasicLayerConfig` with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `device`: Backend device.
    #[must_use]
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> StochasticDepthTransformerBlockSequence<B> {
        StochasticDepthTransformerBlockSequence {
            blocks: self
                .block_configs()
                .into_iter()
                .map(|config| config.init(device))
                .collect(),
        }
    }
}

/// SWIN-Transformer Transformer Block stack.
///
/// Equivalent to ``BasicLayer`` in the original SWIN-Transformer source.
///
/// Applies a sequence of shift-window-alternating ``SwinTransformerBlock`` modules.
#[derive(Module, Debug)]
pub struct StochasticDepthTransformerBlockSequence<B: Backend> {
    blocks: Vec<ShiftedWindowTransformerBlock<B>>,
}

impl<B: Backend> StochasticDepthTransformerBlockSequenceMeta
    for StochasticDepthTransformerBlockSequence<B>
{
    fn d_input(&self) -> usize {
        self.blocks[0].d_input()
    }

    fn input_resolution(&self) -> [usize; 2] {
        self.blocks[0].input_resolution()
    }

    fn depth(&self) -> usize {
        self.blocks.len()
    }

    fn num_heads(&self) -> usize {
        self.blocks[0].num_heads()
    }

    fn window_size(&self) -> usize {
        self.blocks[0].window_size()
    }

    fn mlp_ratio(&self) -> f64 {
        self.blocks[0].mlp_ratio()
    }

    fn enable_qkv_bias(&self) -> bool {
        self.blocks[0].enable_qkv_bias()
    }

    fn drop_rate(&self) -> f64 {
        self.blocks[0].drop_rate()
    }

    fn attn_drop_rate(&self) -> f64 {
        self.blocks[0].attn_drop_rate()
    }

    fn drop_path_rates(&self) -> Vec<f64> {
        self.blocks.iter().map(|b| b.drop_path_rate()).collect()
    }
}

impl<B: Backend> StochasticDepthTransformerBlockSequence<B> {
    /// Applies the layer to the input tensor.
    ///
    /// # Arguments
    ///
    /// - `x`: Input tensor of shape (B, H * W, C).
    ///
    /// # Returns
    ///
    /// Output tensor of shape (B, H * W, C).
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
    ) -> Tensor<B, 3> {
        let mut x = x;
        for block in &self.blocks {
            x = block.forward(x);
        }
        x
    }
}
