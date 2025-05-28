use crate::models::swin::v2::layer::BasicLayer;
use crate::models::swin::v2::patch::{PatchEmbed, PatchEmbedConfig, PatchEmbedMeta};
use burn::config::Config;
use burn::module::{Module, Param};
use burn::nn::pool::{AdaptiveAvgPool1d, AdaptiveAvgPool1dConfig};
use burn::nn::{
    Dropout, DropoutConfig, Initializer, LayerNorm, LayerNormConfig, Linear, LinearConfig,
};
use burn::prelude::{Backend, Tensor};

#[derive(Config, Debug)]
pub struct LayerConfig {
    pub depth: usize,
    pub num_heads: usize,
}

pub trait SwinTransformerV2Meta {
    /// The input image resolution as [height, width].
    fn input_resolution(&self) -> [usize; 2];

    /// The input height of the image.
    fn input_height(&self) -> usize {
        self.input_resolution()[0]
    }

    /// The input width of the image.
    fn input_width(&self) -> usize {
        self.input_resolution()[1]
    }

    /// The patch size of the input image.
    fn patch_size(&self) -> usize;

    /// The number of input channels.
    fn d_input(&self) -> usize;

    /// Number of classes.
    fn num_classes(&self) -> usize;

    /// The size of the embedding dimension.
    fn d_embed(&self) -> usize;

    /// The window size for window attention.
    fn window_size(&self) -> usize;

    /// Depth of each layer.
    fn layer_configs(&self) -> Vec<LayerConfig>;

    /// Ratio of hidden dimension to input dimension in MLP.
    fn mlp_ratio(&self) -> f64;

    /// Whether to enable QKV bias.
    fn enable_qkv_bias(&self) -> bool;

    /// Dropout rate for MLP.
    fn drop_rate(&self) -> f64;

    /// Dropout rate for attention.
    fn attn_drop_rate(&self) -> f64;

    /// Drop path rate for stochastic depth.
    fn drop_path_rate(&self) -> f64;

    /// Enable APE (Absolute Positional Encoding).
    fn enable_ape(&self) -> bool;

    /// Enable patch normalization?
    fn enable_patch_norm(&self) -> bool;
}

#[derive(Config, Debug)]
pub struct SwinTransformerV2Config {
    /// The input image resolution as [height, width].
    pub input_resolution: [usize; 2],

    /// The patch size of the input image.
    pub patch_size: usize,

    /// The number of input channels.
    pub d_input: usize,

    /// Number of classes.
    pub num_classes: usize,

    /// The size of the embedding dimension.
    pub d_embed: usize,

    /// Depth of each layer.
    pub layer_configs: Vec<LayerConfig>,

    /// The window size for window attention.
    #[config(default = 7)]
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

    /// Drop path rate for stochastic depth.
    #[config(default = 0.1)]
    pub drop_path_rate: f64,

    /// Enable APE (Absolute Positional Encoding).
    #[config(default = true)]
    pub enable_ape: bool,

    /// Enable patch normalization?
    #[config(default = true)]
    pub enable_patch_norm: bool,
}

impl SwinTransformerV2Meta for SwinTransformerV2Config {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.d_input
    }

    fn num_classes(&self) -> usize {
        self.num_classes
    }

    fn d_embed(&self) -> usize {
        self.d_embed
    }

    fn window_size(&self) -> usize {
        self.window_size
    }

    fn layer_configs(&self) -> Vec<LayerConfig> {
        self.layer_configs.clone()
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

    fn drop_path_rate(&self) -> f64 {
        self.drop_path_rate
    }

    fn enable_ape(&self) -> bool {
        self.enable_ape
    }

    fn enable_patch_norm(&self) -> bool {
        self.enable_patch_norm
    }
}

impl SwinTransformerV2Config {
    /// Initialize a new [SwinTransformerV2](SwinTransformerV2) model.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> SwinTransformerV2<B> {
        let patch_embed: PatchEmbed<B> = PatchEmbedConfig::new(
            self.input_resolution,
            self.patch_size,
            self.d_input,
            self.d_embed,
        )
        .with_enable_patch_norm(self.enable_patch_norm)
        .init(device);

        let num_patches = patch_embed.num_patches();

        // ape: trunc_normal: ([1, num_patches, d_embed], std=0.02)
        // defaults: (mean=0.0, a=-2.0, b=2.0)
        let ape: Option<Param<Tensor<B, 3>>> = if self.enable_ape {
            Some(
                Initializer::Normal {
                    mean: 0.0,
                    std: 0.02,
                }
                .init([1_usize, num_patches, self.d_embed], device),
            )
        } else {
            None
        };

        let pos_drop = DropoutConfig::new(self.drop_rate).init();

        let norm: LayerNorm<B> = LayerNormConfig::new(self.d_embed).init(device);

        let avgpool = AdaptiveAvgPool1dConfig::new(1).init();
        let head: Linear<B> = LinearConfig::new(self.d_embed, self.num_classes).init(device);

        unimplemented!()
    }
}

#[derive(Module, Debug)]
pub struct SwinTransformerV2<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub ape: Option<Param<Tensor<B, 3>>>,
    pub pos_drop: Dropout,
    pub layers: Vec<BasicLayer<B>>,
    pub norm: LayerNorm<B>,
    pub avgpool: AdaptiveAvgPool1d,
    pub head: Linear<B>,
}
