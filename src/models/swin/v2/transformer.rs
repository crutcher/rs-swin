use crate::models::swin::v2::dpr::DropPathRateDepthTable;
use crate::models::swin::v2::layer::{BasicLayer, BasicLayerConfig, BasicLayerMeta};
use crate::models::swin::v2::patch::{
    PatchEmbed, PatchEmbedConfig, PatchEmbedMeta, PatchMerging, PatchMergingConfig,
    PatchMergingMeta,
};
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

    /// The number of input channels.
    fn d_input(&self) -> usize;

    /// The patch size of the input image.
    fn patch_size(&self) -> usize;

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

#[derive(Debug)]
pub struct SwinTransformerV2Plan {
    pub patch_config: PatchEmbedConfig,

    pub layer_resolutions: Vec<[usize; 2]>,
    pub layer_dims: Vec<usize>,
}

impl SwinTransformerV2Config {
    /// Check config validity and return a plan for the Swin Transformer V2 model.
    ///
    /// Performs model constraint validation tests without initializing a model.
    ///
    /// # Returns
    ///
    /// A [SwinTransformerV2Plan] containing the patch embedding configuration.
    pub fn validate(&self) -> Result<SwinTransformerV2Plan, String> {
        let patch_config = PatchEmbedConfig::new(
            self.input_resolution,
            self.patch_size,
            self.d_input,
            self.d_embed,
        )
        .with_enable_patch_norm(self.enable_patch_norm);

        if self.layer_configs.is_empty() {
            return Err("At least one layer configuration is required".to_string());
        }

        let mut layer_resolutions: Vec<[usize; 2]> = Vec::with_capacity(self.layer_configs.len());
        let mut layer_dims: Vec<usize> = Vec::with_capacity(self.layer_configs.len());

        for layer_i in 0..self.layer_configs.len() {
            let layer_p = 2usize.pow(layer_i as u32); // Power of 2 for each layer
            layer_resolutions.push([
                patch_config.patches_height() / layer_p,
                patch_config.patches_width() / layer_p,
            ]);
            layer_dims.push(patch_config.d_output() * layer_p);
        }

        Ok(SwinTransformerV2Plan {
            patch_config,
            layer_resolutions,
            layer_dims,
        })
    }

    /// Initialize a new [SwinTransformerV2](SwinTransformerV2) model.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> SwinTransformerV2<B> {
        let plan = self.validate().unwrap();
        println!("plan: {:#?}", plan);

        let patch_embed: PatchEmbed<B> = plan.patch_config.init(device);

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

        let depths = self
            .layer_configs
            .iter()
            .map(|c| c.depth)
            .collect::<Vec<usize>>();
        let num_heads = self
            .layer_configs
            .iter()
            .map(|c| c.num_heads)
            .collect::<Vec<usize>>();

        // Stochastic depth delay rule
        let dpr_layer_rates = DropPathRateDepthTable::dpr_layer_rates(self.drop_path_rate, &depths);

        let layer_stack_blocks: Vec<BasicLayer<B>> = (0..self.layer_configs.len())
            .map(|layer_i| {
                let layer_resolution = plan.layer_resolutions[layer_i];
                let layer_dim = plan.layer_dims[layer_i];

                let config = BasicLayerConfig::new(
                    // Double the embedding size for each layer
                    layer_dim,
                    layer_resolution,
                    depths[layer_i],
                    num_heads[layer_i],
                    self.window_size,
                )
                .with_mlp_ratio(self.mlp_ratio)
                .with_enable_qkv_bias(self.enable_qkv_bias)
                .with_drop_path_rates(Some(dpr_layer_rates[layer_i].clone()));

                config.init::<B>(device)
            })
            .collect();

        let layer_stack_downsamples: Vec<PatchMerging<B>> = (0..self.layer_configs.len() - 1)
            .map(|layer_i| {
                let basic_layer = &layer_stack_blocks[layer_i];

                PatchMergingConfig::new(basic_layer.input_resolution(), basic_layer.d_input())
                    .init(device)
            })
            .collect();

        let layer_stack_output_features = *plan.layer_dims.last().unwrap();

        let layer_stack_norm: LayerNorm<B> =
            LayerNormConfig::new(layer_stack_output_features).init(device);

        let avgpool = AdaptiveAvgPool1dConfig::new(1).init();
        let head: Linear<B> =
            LinearConfig::new(layer_stack_output_features, self.num_classes).init(device);

        println!();
        println!("input_resolution: {:?}", self.input_resolution);
        println!("patch_size: {:?}", self.patch_size);
        println!("embed_resolution: {:?}", patch_embed.input_resolution());
        println!("window_size: {:?}", self.window_size);
        for (i, cfg) in self.layer_configs.iter().enumerate() {
            println!("Layer {i}: {:?}", cfg);

            println!("block: {:?}", layer_stack_blocks[i].input_resolution());

            if i < self.layer_configs.len() - 1 {
                println!(
                    "merge\n - I: {:?}",
                    layer_stack_downsamples[i].input_resolution()
                );
                println!(
                    "merge\n - O: {:?}",
                    layer_stack_downsamples[i].output_resolution()
                );
            }
        }

        SwinTransformerV2 {
            patch_embed,
            ape,
            pos_drop,
            layer_stack_blocks,
            layer_stack_downsamples,
            layer_stack_norm,
            avgpool,
            head,
        }
    }
}

#[derive(Module, Debug)]
pub struct SwinTransformerV2<B: Backend> {
    pub patch_embed: PatchEmbed<B>,
    pub ape: Option<Param<Tensor<B, 3>>>,
    pub pos_drop: Dropout,
    pub layer_stack_blocks: Vec<BasicLayer<B>>,
    pub layer_stack_downsamples: Vec<PatchMerging<B>>,
    pub layer_stack_norm: LayerNorm<B>,
    pub avgpool: AdaptiveAvgPool1d,
    pub head: Linear<B>,
}

impl<B: Backend> SwinTransformerV2<B> {
    /// Applies the model to the input image tensor and returns the patch features.
    ///
    /// This method computes the spatial features from the input image tensor.
    ///
    /// # Arguments
    ///
    /// * `input`: A 4D tensor representing the input image with shape `[B, C, H, W]`,
    ///
    /// # Returnso
    ///
    /// A 3D tensor of shape `[B, L, C]` representing the spatial features extracted from the input image.
    pub fn patch_features(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        let mut x = self.patch_embed.forward(input);

        x = if self.ape.is_some() {
            let ape = self.ape.as_ref().unwrap();
            x + ape.val()
        } else {
            x
        };

        x = self.pos_drop.forward(x);

        for layer_i in 0..self.layer_stack_blocks.len() {
            if layer_i > 0 {
                x = self.layer_stack_downsamples[layer_i - 1].forward(x);
            }

            x = self.layer_stack_blocks[layer_i].forward(x);
        }
        // B L C

        x = self.layer_stack_norm.forward(x);
        // B L C

        x
    }

    /// Applies the model without the final classification head.
    ///
    /// This method computes the features from the input image tensor.
    ///
    /// # Arguments
    ///
    /// * `input`: A 4D tensor representing the input image with shape `[B, C, H, W]`,
    ///
    /// # Returns
    ///
    /// A 2D tensor of shape `[B, C]` representing the features extracted from the input image.
    pub fn forward_features(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let x = self.patch_features(input);
        // B L C

        let x = x.swap_dims(1, 2);
        // B C L

        let x = self.avgpool.forward(x);
        // B C 1

        x.flatten::<2>(1, 2)
        // B C
    }

    /// Applies the model to the input image tensor and returns the classification logits.
    ///
    /// # Arguments
    ///
    /// * `input`: A 4D tensor representing the input image with shape `[B, C, H, W]`,
    ///
    /// # Returns
    ///
    /// A 2D tensor of shape `[B, num_classes]` representing the classification logits.
    pub fn forward(
        &self,
        input: Tensor<B, 4>,
    ) -> Tensor<B, 2> {
        let features = self.forward_features(input);
        self.head.forward(features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;

    #[test]
    fn test_swin_transformer_v2_config() {
        let config = SwinTransformerV2Config {
            input_resolution: [224, 224],
            patch_size: 4,
            d_input: 3,
            num_classes: 1000,
            d_embed: 96,
            layer_configs: vec![
                LayerConfig {
                    depth: 2,
                    num_heads: 3,
                },
                LayerConfig {
                    depth: 2,
                    num_heads: 6,
                },
                LayerConfig {
                    depth: 18,
                    num_heads: 12,
                },
            ],
            window_size: 7,
            mlp_ratio: 4.0,
            enable_qkv_bias: true,
            drop_rate: 0.0,
            attn_drop_rate: 0.0,
            drop_path_rate: 0.1,
            enable_ape: true,
            enable_patch_norm: true,
        };

        assert_eq!(config.input_resolution, [224, 224]);
        assert_eq!(config.patch_size, 4);
        assert_eq!(config.d_input, 3);
        assert_eq!(config.num_classes, 1000);
        assert_eq!(config.d_embed, 96);
        assert_eq!(config.window_size, 7);
        assert_eq!(config.mlp_ratio, 4.0);
        assert!(config.enable_qkv_bias);
        assert_eq!(config.drop_rate, 0.0);
        assert_eq!(config.attn_drop_rate, 0.0);
        assert_eq!(config.drop_path_rate, 0.1);
        assert!(config.enable_ape);
        assert!(config.enable_patch_norm);
    }

    #[test]
    #[allow(unused_variables)]
    fn test_forward() {
        let b = 2;
        let d_input = 3;

        let layer_configs = vec![
            LayerConfig {
                depth: 2,
                num_heads: 3,
            },
            LayerConfig {
                depth: 2,
                num_heads: 6,
            },
            LayerConfig {
                depth: 6,
                num_heads: 12,
            },
            LayerConfig {
                depth: 2,
                num_heads: 24,
            },
        ];
        let k = layer_configs.len() - 1;

        let patch_size = 4;
        let window_size = 7;

        let last_wh = 2;
        let last_ww = 2;
        let last_h = last_wh * window_size;
        let last_w = last_ww * window_size;
        let h = last_h * 2usize.pow(k as u32) * patch_size;
        let w = last_w * 2usize.pow(k as u32) * patch_size;

        // h: 3 * 7(w) * 2(m) * 2(m) * 2(m) * 4(patch)

        let num_classes = 12;

        let d_embed = (d_input * patch_size * patch_size) / 2;

        let config = SwinTransformerV2Config::new(
            [h, w],
            patch_size,
            d_input,
            num_classes,
            d_embed,
            layer_configs,
        )
        .with_window_size(window_size);

        let device = Default::default();
        let model: SwinTransformerV2<NdArray> = config.init(&device);

        // assert!(false);

        let distribution = Distribution::Normal(0.0, 0.02);
        let input = Tensor::<NdArray, 4>::random([b, d_input, h, w], distribution, &device);

        let output = model.forward(input);
    }
}
