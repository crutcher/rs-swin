//! # Tensor Clamping Support

use burn::prelude::{Backend, Tensor};
use serde::{Deserialize, Serialize};

/// Configuration for clamping.
#[derive(Default, Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ClampConfig {
    /// The minimum value.
    min: Option<f64>,

    /// The maximum value.
    max: Option<f64>,
}

impl ClampConfig {
    /// Extend the clamp with a minimum value.
    pub fn with_min(
        self,
        min: f64,
    ) -> Self {
        Self {
            min: Some(min),
            ..self
        }
    }

    /// Extend the clamp with a maximum value.
    pub fn with_max(
        self,
        max: f64,
    ) -> Self {
        Self {
            max: Some(max),
            ..self
        }
    }

    /// Apply the clamp.
    pub fn clamp<B: Backend, const D: usize>(
        &self,
        tensor: Tensor<B, D>,
    ) -> Tensor<B, D> {
        match (self.min, self.max) {
            (Some(min), Some(max)) => tensor.clamp(min, max),
            (Some(min), None) => tensor.clamp_min(min),
            (None, Some(max)) => tensor.clamp_max(max),
            (None, None) => tensor,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::TensorData;
    use num_traits::clamp;

    #[test]
    fn test_config() {
        type B = NdArray;
        let device = Default::default();

        let cfg = ClampConfig::default();
        assert_eq!(
            cfg,
            ClampConfig {
                min: None,
                max: None,
            }
        );
        let tensor = Tensor::<B, 1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-1.0, 0.0, 1.0]), false);

        let cfg = ClampConfig::default().with_min(-0.5).with_max(0.5);
        assert_eq!(
            cfg,
            ClampConfig {
                min: Some(-0.5),
                max: Some(0.5),
            }
        );
        let tensor = Tensor::<B, 1>::from_data([-1.0, 0.0, 1.0], &device);
        let tensor = cfg.clamp(tensor);
        tensor
            .to_data()
            .assert_eq(&TensorData::from([-0.5, 0.0, 0.5]), false);
    }
}
