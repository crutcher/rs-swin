use bimm_contracts_shapes::{DimExpr, DimMatcher, ShapeContract};
use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{LayerNorm, LayerNormConfig};
use burn::prelude::{Backend, Tensor};

/// Common introspection interface for PatchEmbed modules.
pub trait PatchEmbedMeta {
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

    /// Input feature dimension size.
    fn d_input(&self) -> usize;

    /// The size of each patch.
    fn patch_size(&self) -> usize;

    /// Image resolution, measured in patches.
    fn patches_resolution(&self) -> [usize; 2] {
        [self.patches_height(), self.patches_width()]
    }

    /// Height of the image, measured in patches.
    fn patches_height(&self) -> usize {
        self.input_height() / self.patch_size()
    }

    /// Width of the image, measured in patches.
    fn patches_width(&self) -> usize {
        self.input_width() / self.patch_size()
    }

    /// Total number of patches.
    fn num_patches(&self) -> usize {
        self.patches_height() * self.patches_width()
    }

    /// Output feature dimension size.
    fn d_output(&self) -> usize {
        self.d_input()
    }

    /// Enable patch normalization.
    fn enable_patch_norm(&self) -> bool;
}

/// Configuration for PatchEmbed.
#[derive(Config, Debug, Copy)]
pub struct PatchEmbedConfig {
    /// Input resolution (height, width).
    input_resolution: [usize; 2],

    /// Patch size.
    patch_size: usize,

    /// Input feature dimension size.
    d_input: usize,

    /// Output feature dimension size.
    d_output: usize,

    /// Enable patch normalization.
    #[config(default = true)]
    enable_patch_norm: bool,
}

impl PatchEmbedMeta for PatchEmbedConfig {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_output(&self) -> usize {
        self.d_output
    }

    fn enable_patch_norm(&self) -> bool {
        self.enable_patch_norm
    }
}

impl PatchEmbedConfig {
    /// Initialize a PatchEmbed module with the given configuration.
    ///
    /// ## Arguments
    ///
    /// * `device` - The device on which the module will be initialized.
    ///
    /// ## Returns
    ///
    /// * A `PatchEmbed` module configured with the specified parameters.
    #[must_use]
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> PatchEmbed<B> {
        let [h, w] = self.input_resolution;
        assert!(
            h % self.patch_size == 0 && w % self.patch_size == 0,
            "Input resolution must be divisible by patch size: {:?}",
            self.input_resolution
        );

        let stride = [self.patch_size, self.patch_size];

        PatchEmbed {
            input_resolution: self.input_resolution,
            patch_size: self.patch_size,
            projection: Conv2dConfig::new([self.d_input, self.d_output], stride)
                .with_stride(stride)
                .init(device),
            norm: match self.enable_patch_norm {
                true => Some(LayerNormConfig::new(self.d_output()).init(device)),
                false => None,
            },
        }
    }
}

/// SWIN-Transformer v2 PatchEmbed module.
#[derive(Module, Debug)]
pub struct PatchEmbed<B: Backend> {
    pub input_resolution: [usize; 2],
    pub patch_size: usize,
    pub projection: Conv2d<B>,
    pub norm: Option<LayerNorm<B>>,
}

impl<B: Backend> PatchEmbedMeta for PatchEmbed<B> {
    fn input_resolution(&self) -> [usize; 2] {
        self.input_resolution
    }

    fn patch_size(&self) -> usize {
        self.patch_size
    }

    fn d_input(&self) -> usize {
        self.projection.weight.dims()[1]
    }

    fn d_output(&self) -> usize {
        self.projection.weight.dims()[0]
    }

    fn enable_patch_norm(&self) -> bool {
        self.norm.is_some()
    }
}

impl<B: Backend> PatchEmbed<B> {
    /// Apply the PatchEmbed module to an input tensor.
    ///
    /// ## Arguments
    ///
    /// * `x` - Input tensor of shape ``(B, C, H, W)``.
    ///
    /// ## Returns
    ///
    /// * Output tensor of shape ``(B, H/patch_size * W/patch_size, d_output)``.
    #[must_use]
    pub fn forward(
        &self,
        x: Tensor<B, 4>,
    ) -> Tensor<B, 3> {
        static CONTRACT: ShapeContract = ShapeContract::new(&[
            DimMatcher::Expr(DimExpr::Param("batch")),
            DimMatcher::Expr(DimExpr::Param("channels")),
            DimMatcher::Expr(DimExpr::Param("height")),
            DimMatcher::Expr(DimExpr::Param("width")),
        ]);
        CONTRACT.assert_shape(
            &x.dims(),
            &[
                ("height", self.input_height()),
                ("width", self.input_width()),
            ],
        );

        let x = self.projection.forward(x);
        let x = x.flatten(2, 3);
        let x = x.swap_dims(1, 2);

        match self.norm {
            None => x,
            Some(ref norm) => norm.forward(x),
        }
    }
}
