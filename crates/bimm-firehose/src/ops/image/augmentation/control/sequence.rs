use crate::define_image_aug_plugin;
use crate::ops::image::augmentation::plugins::{
    AugmentationStage, AugmentationStageConfig, ImageAugContext, PluginBuilder, WithPluginBuilder,
};
use image::DynamicImage;
use serde::{Deserialize, Serialize};
use serde_json::json;

define_image_aug_plugin!(SEQUENCE, SequencePlugin::build_plugin);

/// Serialized configuration for the `SequencePlugin`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SequencePluginConfig {
    /// The sequence of augmentation stages.
    pub stages: Vec<AugmentationStageConfig>,
}

/// A plugin which applies a sequence of augmentation stages.
#[derive(Debug, Clone)]
pub struct SequencePlugin {
    /// The sequence of augmentation stages.
    pub stages: Vec<Box<dyn AugmentationStage>>,
}

impl WithPluginBuilder for SequencePlugin {
    /// Builder for `SequencePlugin`.
    fn build_plugin(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Box<dyn AugmentationStage>> {
        let config: SequencePluginConfig = serde_json::from_value(config.body.clone())?;
        let stages = builder.build_stage_vector(&config.stages)?;
        Ok(Box::new(SequencePlugin { stages }))
    }
}

impl AugmentationStage for SequencePlugin {
    fn name(&self) -> &str {
        SEQUENCE
    }

    fn as_config_body(&self) -> serde_json::Value {
        json! {{
            "stages": self.stages.iter().map(|s| s.as_config()).collect::<Vec<_>>(),
        }}
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        let mut image = image;
        for stage in &self.stages {
            image = stage.augment_image(image, ctx)?;
        }
        Ok(image)
    }
}
