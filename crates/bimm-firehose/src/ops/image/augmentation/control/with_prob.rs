use crate::define_image_aug_plugin;
use crate::ops::image::augmentation::{
    AugmentationStage, AugmentationStageConfig, ImageAugContext, PluginBuilder,
    WithAugmentationStageBuilder,
};
use crate::utility::probability::try_probability;
use image::DynamicImage;
use rand::Rng;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

define_image_aug_plugin!(WITH_PROB, WithProbStage::build_stage);

/// Serialized configuration for the `WithProbStage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WithProbStageConfig {
    /// The probability of applying the inner stage.
    pub prob: f64,

    /// The inner stage.
    pub inner: AugmentationStageConfig,
}

/// A plugin which stochastically applies an inner stage.
#[derive(Debug, Clone)]
pub struct WithProbStage {
    /// The probability of applying the inner stage.
    prob: f64,

    /// The wrapped stage.
    inner: Arc<dyn AugmentationStage>,
}

impl WithAugmentationStageBuilder for WithProbStage {
    /// Builder for `WithProbStage`.
    fn build_stage(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Arc<dyn AugmentationStage>> {
        let config: WithProbStageConfig = serde_json::from_value(config.body.clone())?;

        Ok(Arc::new(Self {
            prob: try_probability(config.prob)?,
            inner: builder.build_stage(&config.inner)?,
        }))
    }
}

impl AugmentationStage for WithProbStage {
    fn name(&self) -> &str {
        WITH_PROB
    }

    fn as_config_body(&self) -> serde_json::Value {
        serde_json::to_value(WithProbStageConfig {
            prob: self.prob,
            inner: self.inner.as_config(),
        })
        .unwrap()
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        let mut image = image;
        if ctx.rng_mut().random::<f64>() < self.prob {
            image = self.inner.augment_image(image, ctx)?;
        }
        Ok(image)
    }
}
