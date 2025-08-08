use image::DynamicImage;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
/// Defines an ID and registers a [`AugmentationStage`].
#[macro_export]
macro_rules! define_image_aug_plugin {
    ($name:ident, $builder:expr) => {
        $crate::define_image_aug_plugin_id!($name);
        $crate::register_image_aug_plugin!($name, $builder);
    };
}

/// Registers a [`AugmentationStage`].
///
/// ## Arguments
///
/// - `name`: the local reference name of the plugin ID;
///   used by `define_image_aug_plugin!()`.
/// - `builder`: the plugin builder operation.
#[macro_export]
macro_rules! register_image_aug_plugin {
    ($name:ident, $builder:expr) => {
        inventory::submit! {
            $crate::ops::image::augmentation::plugins::AugmentationStageGlobalRegistration {
                name: $name,
                build_stage: |cfg, builder| ($builder)(cfg, builder),
            }
        }
    };
}

/// Defines a self-referential plugin ID for [`AugmentationStage`].
#[macro_export]
macro_rules! define_image_aug_plugin_id {
    ($name:ident) => {
        $crate::define_self_referential_id!($name);
    };
}

/// Registration record for [`AugmentationStage`] builders.
pub struct AugmentationStageGlobalRegistration {
    /// The plugin ID.
    pub name: &'static str,

    /// The plugin builder.
    pub build_stage: fn(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Box<dyn AugmentationStage>>,
}
inventory::collect!(AugmentationStageGlobalRegistration);

/// Builder for the `PluginConfig` to `ImageAugPlugin` path.
pub trait PluginBuilder {
    /// Build an `AugmentationStage` from config.
    fn build_stage(
        &self,
        config: &AugmentationStageConfig,
    ) -> anyhow::Result<Box<dyn AugmentationStage>>;

    /// Build a vector of
    fn build_stage_vector(
        &self,
        configs: &Vec<AugmentationStageConfig>,
    ) -> anyhow::Result<Vec<Box<dyn AugmentationStage>>> {
        let mut plugins = Vec::with_capacity(configs.len());
        for config in configs {
            plugins.push(self.build_stage(config)?);
        }
        Ok(plugins)
    }
}

/// Trait defining an associated builder for a plugin.
pub trait WithPluginBuilder {
    /// Build a plugin.
    fn build_plugin(
        config: &AugmentationStageConfig,
        builder: &dyn PluginBuilder,
    ) -> anyhow::Result<Box<dyn AugmentationStage>>;
}

/// Global plugin registry builder.
pub struct GlobalRegistryBuilder;

impl PluginBuilder for GlobalRegistryBuilder {
    fn build_stage(
        &self,
        config: &AugmentationStageConfig,
    ) -> anyhow::Result<Box<dyn AugmentationStage>> {
        let name = config.name.as_str();

        let reg = inventory::iter::<AugmentationStageGlobalRegistration>
            .into_iter()
            .find(|reg| reg.name == name)
            .ok_or_else(|| anyhow::anyhow!("No plugin named {}", name))?;

        (reg.build_stage)(config, self)
    }
}

/// Augmentation context, used by `AugmentationStage::augment_image`.
#[derive(Debug, Clone)]
pub struct ImageAugContext {
    /// The context random number generator.
    rng: rand::rngs::StdRng,
}

impl ImageAugContext {
    /// Construct a new context.
    pub fn new(rng: rand::rngs::StdRng) -> Self {
        Self { rng }
    }

    /// Get a mutable reference to the context's random number generator.
    pub fn rng_mut(&mut self) -> &mut rand::rngs::StdRng {
        &mut self.rng
    }
}

/// Trait to help `Box<dyn AugmentationStep>` implement Clone.
///
/// This is helpful for nesting.
pub trait CloneBox {
    /// Clone the plugin into a Box.
    ///
    /// This is used with an impl of `Clone` for `Box<dyn ImagePlugin>`
    /// to make plugins containable.
    fn clone_box(&self) -> Box<dyn AugmentationStage>;
}

/// A trait defining a plugin for image augmentation.
pub trait AugmentationStage: Debug + CloneBox {
    /// Get the stage name.
    fn name(&self) -> &str;

    /// Construct the body of a config for this stage.
    fn as_config_body(&self) -> serde_json::Value;

    /// Construct a config for this.
    fn as_config(&self) -> AugmentationStageConfig {
        AugmentationStageConfig {
            name: self.name().to_string(),
            body: self.as_config_body(),
        }
    }

    /// Apply the stage to the image.
    ///
    /// ## Arguments
    ///
    /// - `image`: the image to augment.
    /// - `ctx`: the `AugmentationContex` being operated in.
    ///
    /// ## Returns
    ///
    /// A (modified?) image.
    fn augment_image(
        &self,
        image: DynamicImage,
        ctx: &mut ImageAugContext,
    ) -> anyhow::Result<DynamicImage>;
}

impl<T> CloneBox for T
where
    T: 'static + AugmentationStage + Clone,
{
    fn clone_box(&self) -> Box<dyn AugmentationStage> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn AugmentationStage> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

/// Serializable config for name + body for `AugmentationStage`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AugmentationStageConfig {
    /// The plugin name.
    pub name: String,

    /// The body of the plugin.
    #[serde(default, skip_serializing_if = "serde_json::Value::is_null")]
    pub body: serde_json::Value,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ops::image::augmentation::control::noop::NOOP;
    use crate::ops::image::augmentation::control::sequence::SEQUENCE;
    use serde_json::json;

    #[test]
    fn test_plugin_builder() -> anyhow::Result<()> {
        let cfg_json = json! {
            {
                "name": SEQUENCE,
                "body": {
                  "stages": [
                    {
                      "name": NOOP,
                    }
                  ]
                }
            }
        };
        let pretty = serde_json::to_string_pretty(&cfg_json)?;
        println!("{pretty}");

        let cfg = serde_json::from_value::<AugmentationStageConfig>(cfg_json.clone())?;

        let builder = GlobalRegistryBuilder {};

        let plugin = builder.build_stage(&cfg)?;

        let new_cfg = plugin.as_config();
        let new_pretty = serde_json::to_string_pretty(&new_cfg)?;
        println!("{new_pretty}");

        // assert!(false);

        Ok(())
    }
}
