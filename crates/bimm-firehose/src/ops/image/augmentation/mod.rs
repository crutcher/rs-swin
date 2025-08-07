use crate::core::operations::factory::SimpleConfigOperatorFactory;
use crate::core::operations::operator::FirehoseOperator;
use crate::core::operations::planner::OperationPlan;
use crate::core::operations::signature::{FirehoseOperatorSignature, ParameterSpec};
use crate::core::rows::FirehoseRowTransaction;
use crate::core::{FirehoseRowReader, FirehoseRowWriter, FirehoseValue};
use crate::define_firehose_operator;
pub use image::imageops::FilterType;
pub use image::{ColorType, DynamicImage};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};

/// Image augmentation stages.
pub mod stage;

define_firehose_operator!(
    AUG_IMAGE,
    SimpleConfigOperatorFactory::<ImageAugmenter>::new(
        FirehoseOperatorSignature::from_operator_id(AUG_IMAGE)
            .with_description("Loads an image from disk.")
            .with_input(ParameterSpec::new::<u64>("seed").with_description("Augmentation seed."),)
            .with_input(
                ParameterSpec::new::<DynamicImage>("source").with_description("Source image."),
            )
            .with_output(
                ParameterSpec::new::<DynamicImage>("result").with_description("Result image."),
            ),
    )
);

/// Represents the specifications for flipping an image.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FlipSpec {
    /// The probability of flipping the image horizontally.
    pub horizontal: f32,

    /// The probability of flipping the image vertically.
    pub vertical: f32,
}

impl Default for FlipSpec {
    fn default() -> Self {
        Self::new()
    }
}

impl FlipSpec {
    /// Creates a new `FlipSpec` with the specified horizontal and vertical flip probabilities.
    pub fn new() -> Self {
        FlipSpec {
            horizontal: 0.0,
            vertical: 0.0,
        }
    }

    /// Creates a new `FlipSpec` with the specified horizontal flip probabilities.
    pub fn with_horizontal(
        self,
        horizontal: f32,
    ) -> Self {
        FlipSpec {
            horizontal,
            vertical: self.vertical,
        }
    }

    /// Creates a new `FlipSpec` with the specified vertical flip probabilities.
    pub fn with_vertical(
        self,
        vertical: f32,
    ) -> Self {
        FlipSpec {
            horizontal: self.horizontal,
            vertical,
        }
    }
}

/// Image augmentation operator.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageAugmenter {
    /// The flip specifications for the image.
    #[serde(skip_serializing_if = "Option::is_none")]
    #[serde(default)]
    pub flip: Option<FlipSpec>,

    /// Whether to rotate the image.
    #[serde(default)]
    pub rotate: bool,
}

impl Default for ImageAugmenter {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageAugmenter {
    /// Creates a new `ImageLoader` with optional resize and recolor specifications.
    pub fn new() -> Self {
        ImageAugmenter {
            flip: None,
            rotate: false,
        }
    }

    /// Extend the augmenter with a `FlipSpec`.
    pub fn with_flip(
        self,
        flip: FlipSpec,
    ) -> Self {
        ImageAugmenter {
            rotate: self.rotate,
            flip: Some(flip),
        }
    }

    /// Extend the augmenter by enabling/disabling rotate.
    pub fn with_rotate(
        self,
        rotate: bool,
    ) -> Self {
        ImageAugmenter {
            flip: self.flip,
            rotate,
        }
    }

    /// Converts into an `OperationPlanner`
    ///
    /// ## Arguments
    ///
    /// * `seed_column`: The name of the input column containing the augmentation seed.
    /// * `source_column`: The name of the input image column.
    /// * `result_column`: The name of the output image column.
    pub fn to_plan(
        &self,
        seed_column: &str,
        source_column: &str,
        result_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(AUG_IMAGE)
            .with_input("seed", seed_column)
            .with_input("source", source_column)
            .with_output("result", result_column)
            .with_config(self.clone())
    }
}

impl FirehoseOperator for ImageAugmenter {
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        let seed: u64 = txn.maybe_get("seed").unwrap().parse_as()?;
        let mut rng = StdRng::seed_from_u64(seed);

        let source: &DynamicImage = txn.maybe_get("source").unwrap().as_ref()?;

        let mut image = source.clone();

        if let Some(flip) = &self.flip {
            if flip.horizontal > 0.0 && rng.random::<f32>() < flip.horizontal {
                image = image.fliph();
            }
            if flip.vertical > 0.0 && rng.random::<f32>() < flip.vertical {
                image = image.flipv();
            }
        }

        if self.rotate {
            for _ in 0..rng.random_range(0..=3) {
                image = image.rotate90();
            }
        }

        txn.expect_set("result", FirehoseValue::boxing(image));

        Ok(())
    }
}

#[cfg(test)]
mod tests {}
