use image::DynamicImage;
use rand::Rng;
use serde::{Deserialize, Serialize};
use serde_json::json;
use std::fmt::Debug;

/// A stage of image augmentation.
pub trait AugmentationStage: Debug + Send + Sync {
    /// Creates and returns a boxed clone of the current `AugmentationStage` object.
    ///
    /// This method is intended to be implemented by types that implement
    /// the `AugmentationStage` trait. It allows for cloning objects
    /// of types that are stored behind a `Box<dyn AugmentationStage>` pointer,
    /// enabling polymorphic clones of trait objects.
    ///
    /// # Returns
    ///
    /// A `Box` containing a clone of the current `AugmentationStage` object.
    fn clone_box(&self) -> Box<dyn AugmentationStage>;

    fn as_config(&self) -> serde_json::Value {
        let mut obj =
        json! {{
            "_type": self.stage_name(),
        }};

        let inner = self.as_config_inner();
        if !inner.is_null() {
            obj["config"] = inner;
        }

        obj
    }

    fn as_config_inner(&self) -> serde_json::Value;

    fn stage_name(&self) -> String {
        String::from(std::any::type_name::<Self>())
    }

    /// Apply the stage to the image.
    ///
    /// ## Arguments
    ///
    /// - `image`: the image to augment.
    /// - `rng`: a random number generator.
    ///
    /// ## Returns
    ///
    /// A (modified?) image.
    fn apply_stage(
        &self,
        image: DynamicImage,
        rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<DynamicImage>;
}

impl Clone for Box<dyn AugmentationStage> {
    fn clone(&self) -> Self {
        self.clone_box()
    }
}

impl Serialize for Box<dyn AugmentationStage> {
    fn serialize<S>(
        &self,
        serializer: S,
    ) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        self.as_config().serialize(serializer)
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipHorizontally;

impl AugmentationStage for FlipHorizontally {
    fn clone_box(&self) -> Box<dyn AugmentationStage> {
        Box::new(self.clone())
    }

    fn as_config_inner(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }

    fn apply_stage(
        &self,
        image: DynamicImage,
        _rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<DynamicImage> {
        Ok(image.fliph())
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipVertically;

impl AugmentationStage for FlipVertically {
    fn clone_box(&self) -> Box<dyn AugmentationStage> {
        Box::new(self.clone())
    }

    fn as_config_inner(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }
    fn apply_stage(
        &self,
        image: DynamicImage,
        _rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<DynamicImage> {
        Ok(image.flipv())
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct WithProb {
    pub probability: f64,
    pub wrapped: Box<dyn AugmentationStage>,
}

impl AugmentationStage for WithProb {
    fn clone_box(&self) -> Box<dyn AugmentationStage> {
        Box::new(self.clone())
    }

    fn as_config_inner(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }

    fn apply_stage(
        &self,
        image: DynamicImage,
        rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<DynamicImage> {
        if rng.random::<f64>() < self.probability {
            self.wrapped.apply_stage(image, rng)
        } else {
            Ok(image)
        }
    }
}

#[derive(Debug, Clone, Serialize)]
pub struct Choose {
    pub options: Vec<Box<dyn AugmentationStage>>,
}

impl AugmentationStage for Choose {
    fn clone_box(&self) -> Box<dyn AugmentationStage> {
        Box::new(self.clone())
    }

    fn as_config_inner(&self) -> serde_json::Value {
        serde_json::to_value(self).unwrap()
    }

    fn apply_stage(
        &self,
        image: DynamicImage,
        rng: &mut rand::rngs::StdRng,
    ) -> anyhow::Result<DynamicImage> {
        let index = rng.random_range(0..self.options.len());
        let child = &self.options[index];
        child.apply_stage(image, rng)
    }
}

#[cfg(test)]
mod tests {
    use crate::ops::image::augmentation::stage::{AugmentationStage, FlipHorizontally, WithProb};
    use indoc::indoc;

    #[test]
    fn test_config() {
        let stage = WithProb {
            probability: 0.5,
            wrapped: Box::new(FlipHorizontally),
        };

        let config = stage.as_config();
        assert_eq!(
            serde_json::to_string_pretty(&config).unwrap(),
            indoc! {r#"
               {
                 "_type": "bimm_firehose::ops::image::augmentation::stage::WithProb",
                 "config": {
                   "probability": 0.5,
                   "wrapped": {
                     "_type": "bimm_firehose::ops::image::augmentation::stage::FlipHorizontally"
                   }
                 }
               }"#}
        );
    }
}
