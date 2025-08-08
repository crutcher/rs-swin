use crate::define_image_aug_plugin;
use crate::ops::image::augmentation::{
    AugmentationStage, AugmentationStageConfig, ImageAugContext, PluginBuilder,
    WithAugmentationStageBuilder,
};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

define_image_aug_plugin!(HORIZONTAL_FLIP, HorizontalFlipStage::build_stage);

/// An `AugmentationStage` that flips the image horizontally.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct HorizontalFlipStage;

impl WithAugmentationStageBuilder for HorizontalFlipStage {
    fn build_stage(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Box<dyn AugmentationStage>> {
        Ok(Box::new(serde_json::from_value::<HorizontalFlipStage>(
            config.body.clone(),
        )?))
    }
}

impl AugmentationStage for HorizontalFlipStage {
    fn name(&self) -> &str {
        HORIZONTAL_FLIP
    }

    fn as_config_body(&self) -> Value {
        Value::Null
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        _ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        Ok(image.fliph())
    }
}

define_image_aug_plugin!(VERTICAL_FLIP, VerticalFlipStage::build_stage);

/// An `AugmentationStage` that flips the image vertically.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerticalFlipStage;

impl WithAugmentationStageBuilder for VerticalFlipStage {
    fn build_stage(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Box<dyn AugmentationStage>> {
        Ok(Box::new(serde_json::from_value::<VerticalFlipStage>(
            config.body.clone(),
        )?))
    }
}

impl AugmentationStage for VerticalFlipStage {
    fn name(&self) -> &str {
        VERTICAL_FLIP
    }

    fn as_config_body(&self) -> Value {
        Value::Null
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        _ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        Ok(image.flipv())
    }
}
