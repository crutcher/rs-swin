//! # The `ResNet` basic block Implementation.

use burn::config::Config;
use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{BatchNorm, BatchNormConfig, PaddingConfig2d};
use burn::prelude::Backend;

/// `BasicBlock` Config and Model Meta API.
pub trait BasicBlockMeta {
    /// Input channels dimensionality.
    fn in_planes(&self) -> usize;

    /// Used to determine output channels dimensionality.
    fn planes(&self) -> usize;

    /// Stride used in convolution layers.
    fn stride(&self) -> usize;

    /// Reduction factor for the first convolution layer.
    fn reduce_first(&self) -> usize;

    /// The expansion factor for the block.
    fn expansion(&self) -> usize;

    /// The first convolution layer dilation; if set.
    fn first_dilation(&self) -> Option<usize>;

    /// The dilation rate for convolution layers.
    fn dilation(&self) -> usize;

    /// The first convolution layer output channels dimensionality.
    fn first_planes(&self) -> usize {
        self.planes() / self.reduce_first()
    }

    /// The output channels dimensionality.
    fn out_planes(&self) -> usize {
        self.planes() * self.expansion()
    }
}

/// `BasicBlock` Config.
#[derive(Config, Debug)]
pub struct BasicBlockConfig {
    /// Input channels dimensionality.
    pub in_planes: usize,

    /// Used to determine output channels dimensionality.
    pub planes: usize,

    /// Stride used in convolution layers.
    #[config(default = 1)]
    pub stride: usize,

    /// Reduction factor for the first convolution layer.
    #[config(default = 1)]
    pub reduce_first: usize,

    /// The expansion factor for the block.
    #[config(default = 1)]
    pub expansion: usize,

    /// The first convolution layer dilation; if set.
    #[config(default = "Option::None")]
    pub first_dilation: Option<usize>,

    /// The dilation rate for convolution layers.
    #[config(default = 1)]
    pub dilation: usize,
}

impl BasicBlockMeta for BasicBlockConfig {
    fn in_planes(&self) -> usize {
        self.in_planes
    }

    fn planes(&self) -> usize {
        self.planes
    }

    fn stride(&self) -> usize {
        self.stride
    }

    fn reduce_first(&self) -> usize {
        self.reduce_first
    }

    fn expansion(&self) -> usize {
        self.expansion
    }

    fn first_dilation(&self) -> Option<usize> {
        self.first_dilation
    }

    fn dilation(&self) -> usize {
        self.dilation
    }
}

impl BasicBlockConfig {
    /// Initialize `BasicBlock` model.
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> BasicBlock<B> {
        let first_planes = self.first_planes();
        let first_dilation = self.first_dilation.unwrap_or(self.dilation);

        // TODO: aa_layer?

        let conv1_config: Conv2dConfig = Conv2dConfig::new([self.in_planes, first_planes], [3, 3])
            .with_stride([self.stride, self.stride])
            .with_padding(PaddingConfig2d::Explicit(first_dilation, first_dilation))
            .with_bias(false);

        let conv1 = conv1_config.init(device);

        let bn1_config = BatchNormConfig::new(first_planes);
        let bn1: BatchNorm<B, 2> = bn1_config.init(device);

        // let out_planes = self.out_planes();

        BasicBlock { conv1, bn1 }
    }
}

/// `ResNet` `BasicBlock`.
#[derive(Module, Debug)]
pub struct BasicBlock<B: Backend> {
    conv1: Conv2d<B>,
    bn1: BatchNorm<B, 2>,
}

#[cfg(test)]
mod tests {
    use crate::models::resnet::block::{BasicBlockConfig, BasicBlockMeta};

    #[test]
    fn test_basic_block_config() {
        let in_planes = 16;
        let planes = 32;

        let cfg = BasicBlockConfig::new(in_planes, planes);

        assert_eq!(cfg.planes(), planes);
        assert_eq!(cfg.stride(), 1);
    }
}
