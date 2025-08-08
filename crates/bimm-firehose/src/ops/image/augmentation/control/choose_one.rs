use crate::define_image_aug_plugin;
use crate::ops::image::augmentation::{
    AugmentationStage, AugmentationStageConfig, ImageAugContext, PluginBuilder,
    WithAugmentationStageBuilder,
};
use anyhow::bail;
use image::DynamicImage;
use rand::Rng;
use serde::{Deserialize, Serialize};

define_image_aug_plugin!(CHOOSE_ONE, ChooseOneStage::build_stage);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChoiceItemConfig {
    /// The weight of the stage; default, treated as `1.0`.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub weight: Option<f32>,

    /// The augmentation stage.
    pub stage: AugmentationStageConfig,
}

/// Serialized configuration for `StageChoice`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChooseOneStageConfig {
    /// The sequence of augmentation stages.
    pub choices: Vec<ChoiceItemConfig>,
}

/// A stage which applies a sequence of augmentation stages.
#[derive(Debug, Clone)]
pub struct ChooseOneStage {
    /// The weights of the stages.
    weights: Vec<Option<f32>>,

    /// The effective total weight of the stages.
    total_weight: f32,

    /// The sequence of augmentation stages.
    stages: Vec<Box<dyn AugmentationStage>>,
}

impl WithAugmentationStageBuilder for ChooseOneStage {
    /// Builder for `ChoicePlugin`.
    fn build_stage(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Box<dyn AugmentationStage>> {
        let config: ChooseOneStageConfig = serde_json::from_value(config.body.clone())?;

        let mut stages = Vec::with_capacity(config.choices.len());
        let mut weights = Vec::with_capacity(config.choices.len());
        let mut total_weight = 0.0;
        for (idx, choice) in config.choices.iter().enumerate() {
            weights.push(choice.weight);
            let weight = choice.weight.unwrap_or(1.0);
            if weight < 0.0 {
                bail!(
                    "Invalid weight ({}) at index ({idx}):\n{}",
                    weight,
                    serde_json::to_string_pretty(&config)?
                );
            }
            total_weight += weight;

            stages.push(builder.build_stage(&choice.stage)?);
        }

        Ok(Box::new(Self {
            weights,
            total_weight,
            stages,
        }))
    }
}

impl AugmentationStage for ChooseOneStage {
    fn name(&self) -> &str {
        CHOOSE_ONE
    }

    fn as_config_body(&self) -> serde_json::Value {
        serde_json::to_value(ChooseOneStageConfig {
            choices: self
                .stages
                .iter()
                .enumerate()
                .map(|(idx, stage)| ChoiceItemConfig {
                    weight: self.weights[idx],
                    stage: stage.as_config(),
                })
                .collect::<Vec<_>>(),
        })
        .unwrap()
    }

    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage> {
        let mut r = self.total_weight * ctx.rng_mut().random::<f32>();
        let mut idx = 0;
        for weight in &self.weights {
            let weight = weight.unwrap_or(1.0);
            if r < weight {
                break;
            }
            idx += 1;
            r -= weight;
        }
        let stage = &self.stages[idx];
        stage.augment_image(image, ctx)
    }
}
