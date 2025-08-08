use crate::define_image_aug_plugin;
use crate::ops::image::augmentation::plugins::{
    AugmentationStage, AugmentationStageConfig, ImageAugContext, PluginBuilder, WithPluginBuilder,
};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::Value;

define_image_aug_plugin!(NOOP, NoOpPlugin::build_plugin);

/// A no-operation plugin for image augmentation that does nothing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NoOpPlugin;

impl WithPluginBuilder for NoOpPlugin {
    fn build_plugin(
        config: &AugmentationStageConfig,
        _builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Box<dyn AugmentationStage>> {
        Ok(Box::new(serde_json::from_value::<NoOpPlugin>(
            config.body.clone(),
        )?))
    }
}

impl AugmentationStage for NoOpPlugin {
    fn name(&self) -> &str {
        NOOP
    }

    fn as_config_body(&self) -> Value {
        serde_json::Value::Null
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        _ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        Ok(image)
    }
}
