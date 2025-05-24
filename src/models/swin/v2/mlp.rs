use burn::config::Config;
use burn::module::Module;
use burn::nn::{Dropout, DropoutConfig, Gelu, Linear, LinearConfig};
use burn::prelude::{Backend, Tensor};
use burn_contracts::assert_tensor;

pub trait MlpMeta {
    fn d_input(&self) -> usize;

    fn d_hidden(&self) -> usize;

    fn d_output(&self) -> usize;

    fn drop(&self) -> f64;
}

#[derive(Config, Debug)]
pub struct MlpConfig {
    d_input: usize,

    #[config(default = "None")]
    d_hidden: Option<usize>,

    #[config(default = "None")]
    d_output: Option<usize>,

    #[config(default = 0.)]
    drop: f64,
}

impl MlpMeta for MlpConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn d_hidden(&self) -> usize {
        self.d_hidden.unwrap_or(self.d_input)
    }

    fn d_output(&self) -> usize {
        self.d_output.unwrap_or(self.d_input)
    }

    fn drop(&self) -> f64 {
        self.drop
    }
}

impl MlpConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> BlockMlp<B> {
        let d_input = self.d_input();
        let d_hidden = self.d_hidden();
        let d_output = self.d_output();

        BlockMlp {
            fc1: LinearConfig::new(d_input, d_hidden).init(device),
            fc2: LinearConfig::new(d_hidden, d_output).init(device),
            act: Gelu::new(),
            drop: DropoutConfig { prob: self.drop }.init(),
        }
    }
}

/// Swin MLP Module
#[derive(Module, Debug)]
pub struct BlockMlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
    act: Gelu,
    drop: Dropout,
}

impl<B: Backend> MlpMeta for BlockMlp<B> {
    fn d_input(&self) -> usize {
        self.fc1.weight.dims()[0]
    }

    fn d_hidden(&self) -> usize {
        self.fc1.weight.dims()[1]
    }

    fn d_output(&self) -> usize {
        self.fc2.weight.dims()[1]
    }

    fn drop(&self) -> f64 {
        self.drop.prob
    }
}

impl<B: Backend> BlockMlp<B> {
    pub fn forward<const D: usize>(
        &self,
        x: Tensor<B, D>,
    ) -> Tensor<B, D> {
        #[cfg(debug_assertions)]
        assert_tensor(&x)
            .unpacks_shape([], "... in", &[("in", self.d_input())])
            .unwrap();

        let x = self.fc1.forward(x);
        #[cfg(debug_assertions)]
        assert_tensor(&x)
            .unpacks_shape([], "... h", &[("h", self.d_hidden())])
            .unwrap();

        let x = self.act.forward(x);

        let x = self.drop.forward(x);

        let x = self.fc2.forward(x);
        #[cfg(debug_assertions)]
        assert_tensor(&x)
            .unpacks_shape([], "... o", &[("o", self.d_output())])
            .unwrap();

        self.drop.forward(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::tensor::Distribution;
    use burn_contracts::assert_tensor;

    #[test]
    fn test_mlpconfig() {
        {
            let d_input = 4;
            let config = MlpConfig::new(d_input);

            assert_eq!(config.d_input(), d_input);
            assert_eq!(config.d_hidden(), d_input);
            assert_eq!(config.d_output(), d_input);
            assert_eq!(config.drop(), 0.);
        }

        {
            let d_input = 4;
            let d_hidden = 8;
            let d_output = 6;
            let drop = 0.1;

            let config = MlpConfig::new(d_input)
                .with_d_hidden(Some(d_hidden))
                .with_d_output(Some(d_output))
                .with_drop(drop);

            assert_eq!(config.d_input(), d_input);
            assert_eq!(config.d_hidden(), d_hidden);
            assert_eq!(config.d_output(), d_output);
            assert_eq!(config.drop(), drop);
        }
    }

    #[test]
    fn test_mlp() {
        impl_test_mlp::<NdArray>();
    }

    fn impl_test_mlp<B: Backend>() {
        let device: B::Device = Default::default();

        let a = 2;
        let b = 3;
        let d_input = 4;
        let d_hidden = 8;
        let d_output = 6;
        let drop = 0.1;

        let config = MlpConfig::new(d_input)
            .with_d_hidden(Some(d_hidden))
            .with_d_output(Some(d_output))
            .with_drop(drop);

        let mlp: BlockMlp<B> = config.init(&device);

        assert_eq!(mlp.d_input(), config.d_input());
        assert_eq!(mlp.d_hidden(), config.d_hidden());
        assert_eq!(mlp.d_output(), config.d_output());
        assert_eq!(mlp.drop(), config.drop());

        let distribution = Distribution::Normal(0., 1.);
        let x = Tensor::random([a, b, d_input], distribution, &device);

        let y = mlp.forward(x);

        assert_tensor(&y).has_named_dims([("A", a), ("B", b), ("C", d_output)]);
    }
}
