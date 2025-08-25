//! # [`Distribution`] Utility Module

use burn::module::{Content, ModuleDisplay, ModuleDisplayDefault};
use burn::tensor::Distribution;

/// Adapter to display a [`Distribution`] in a module.
pub struct DistributionDisplayAdapter(Distribution);

impl DistributionDisplayAdapter {
    /// Create a new [`DistributionDisplayAdapter`].
    pub fn new(distribution: Distribution) -> Self {
        Self(distribution)
    }
}

impl ModuleDisplay for DistributionDisplayAdapter {}

impl ModuleDisplayDefault for DistributionDisplayAdapter {
    fn content(
        &self,
        content: Content,
    ) -> Option<Content> {
        Some(match self.0 {
            Distribution::Default => content.set_top_level_type("Default"),
            Distribution::Bernoulli(p) => content.set_top_level_type("Bernoulli").add("p", &p),
            Distribution::Uniform(low, high) => content
                .set_top_level_type("Uniform")
                .add("low", &low)
                .add("high", &high),
            Distribution::Normal(mean, std) => content
                .set_top_level_type("Normal")
                .add("mean", &mean)
                .add("std", &std),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::module::DisplaySettings;
    #[test]
    fn test_distribution_display_adapter() {
        let settings = DisplaySettings::default();

        assert_eq!(
            DistributionDisplayAdapter(Distribution::Default).format(settings.clone()),
            "Default",
        );

        assert_eq!(
            DistributionDisplayAdapter(Distribution::Bernoulli(0.5)).format(settings.clone()),
            indoc::indoc! {r#"
                Bernoulli {
                  p: 0.5
                }"#}
        );

        assert_eq!(
            DistributionDisplayAdapter(Distribution::Uniform(0.0, 1.0)).format(settings.clone()),
            indoc::indoc! {r#"
                Uniform {
                  low: 0
                  high: 1
                }"#}
        );

        assert_eq!(
            DistributionDisplayAdapter(Distribution::Normal(0.0, 1.0)).format(settings.clone()),
            indoc::indoc! {r#"
                Normal {
                  mean: 0
                  std: 1
                }"#}
        );
    }
}
