mod grid;
mod mask;
mod rpb;
pub mod swmsa;

pub use grid::*;
pub use mask::*;
pub use rpb::*;

use crate::compat::linalg::l2_normalize;
use burn::config::Config;
use burn::module::{Module, Param, ParamId};
use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::{Backend, Tensor};
use burn::tensor::activation::softmax;
use burn_contracts::assert_tensor;

pub const EPS: f64 = 1e-12;

/// Common introspection interface for WindowAttention.
pub trait WindowAttentionMeta {
    /// Get the input/channel dimension size.
    fn d_input(&self) -> usize;

    /// Get the window size.
    fn window_shape(&self) -> [usize; 2];

    /// Get the number of attention heads.
    fn num_heads(&self) -> usize;

    /// Get the drop rate for attention.
    fn attn_drop(&self) -> f64;

    /// Get the drop rate for projection.
    fn proj_drop(&self) -> f64;

    /// Is the QKV bias enabled?
    fn enable_qkv_bias(&self) -> bool;
}

/// Configuration for the WindowAttention module.
#[derive(Config, Debug)]
pub struct WindowAttentionConfig {
    pub d_input: usize,

    pub window_size: [usize; 2],

    pub num_heads: usize,

    #[config(default = true)]
    pub enable_qkv_bias: bool,

    #[config(default = 0.)]
    pub attn_drop: f64,

    #[config(default = 0.)]
    pub proj_drop: f64,
}

impl WindowAttentionMeta for WindowAttentionConfig {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn window_shape(&self) -> [usize; 2] {
        self.window_size
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn attn_drop(&self) -> f64 {
        self.attn_drop
    }

    fn proj_drop(&self) -> f64 {
        self.proj_drop
    }

    fn enable_qkv_bias(&self) -> bool {
        self.enable_qkv_bias
    }
}

/// The WindowAttention module.
#[derive(Module, Debug)]
pub struct WindowAttention<B: Backend> {
    pub d_input: usize,
    pub num_heads: usize,

    pub q_linear: Linear<B>,
    pub k_linear: Linear<B>,
    pub v_linear: Linear<B>,

    pub logit_scale: Param<Tensor<B, 3>>,
    pub rpb_module: OffsetGridRelativePositionBias<B>,

    pub proj: Linear<B>,

    pub attn_drop: Dropout,
    pub proj_drop: Dropout,
}

impl<B: Backend> WindowAttentionMeta for WindowAttention<B> {
    fn d_input(&self) -> usize {
        self.d_input
    }

    fn window_shape(&self) -> [usize; 2] {
        self.rpb_module.window_shape()
    }

    fn num_heads(&self) -> usize {
        self.num_heads
    }

    fn attn_drop(&self) -> f64 {
        self.attn_drop.prob
    }

    fn proj_drop(&self) -> f64 {
        self.proj_drop.prob
    }

    fn enable_qkv_bias(&self) -> bool {
        self.q_linear.bias.is_some()
    }
}

impl<B: Backend> WindowAttention<B> {
    /// Forward pass of the WindowAttention module.
    ///
    /// ## Arguments
    ///
    /// - `x`: Input tensor of shape (B*num_windows, window_size * window_size, C).
    /// - `mask`: Optional mask tensor of shape (num_windows, Wh*Ww, Wh*Ww).
    ///
    /// ## Returns
    ///
    /// - Output tensor of shape (B*num_windows, N=ws*ws, C).
    pub fn forward(
        &self,
        x: Tensor<B, 3>,
        mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 3> {
        let [b_nw, n, c] = assert_tensor(&x)
            .unpacks_shape(["b_nw", "n", "c"], "b_nw n c", &[("unused", 0)])
            .unwrap();

        // n = ws * ws

        let q = self.q_linear.forward(x.clone());
        let k = self.k_linear.forward(x.clone());
        let v = self.v_linear.forward(x);
        // (b_nw, ws*ws, c)

        let c_per_head = c / self.num_heads;
        let qkv_shape = [b_nw, n, self.num_heads, c_per_head];

        let q = q.reshape(qkv_shape).swap_dims(1, 2);
        let k = k.reshape(qkv_shape).swap_dims(1, 2);
        let v = v.reshape(qkv_shape).swap_dims(1, 2);
        // (b_nw, num_heads, ws*ws, c_per_head)

        let attn = self.attention(b_nw, n, q, k, mask);

        let x = attn.matmul(v);
        let x = x.swap_dims(1, 2).reshape([b_nw, n, c]);
        // (b_nw, ws*ws, c)

        let x = self.proj.forward(x);
        self.proj_drop.forward(x)
        // (b_nw, ws*ws, c)
    }

    /// Compute the attention.
    ///
    /// ## Arguments
    ///
    /// - `b_nw`: Batch size times number of windows.
    /// - `n`: Number of elements in the input tensor.
    /// - `q`: Query tensor of shape (b_nw, num_heads, ws*ws, c_per_head).
    /// - `k`: Key tensor of shape (b_nw, num_heads, ws*ws, c_per_head).
    /// - `mask`: Optional mask tensor of shape (num_windows, ws*ws, ws*ws).
    ///
    /// ## Returns
    ///
    /// - Output attention tensor of shape (b_nw, num_heads, ws*ws, ws*ws).
    fn attention(
        &self,
        b_nw: usize,
        n: usize,
        q: Tensor<B, 4>,
        k: Tensor<B, 4>,
        mask: Option<Tensor<B, 3>>,
    ) -> Tensor<B, 4> {
        // cosine attention
        let q = l2_normalize(q, 3, EPS);
        // (b_nw, num_heads, ws*ws, c_per_head)

        let k = l2_normalize(k, 3, EPS).swap_dims(2, 3);
        // (b_nw, num_heads, c_per_head, ws*ws)

        let attn = q.matmul(k);
        // (b_nw, num_heads, ws*ws, ws*ws)

        let attn = self.encode_attention(attn);
        // (b_nw, num_heads, Wh*Ww, Wh*Ww)

        let attn = self.attn_drop.forward(attn);

        let attn = match mask {
            None => attn,
            Some(mask) => apply_attention_mask(b_nw, n, self.num_heads, attn, mask),
        };

        // (b_nw, num_heads, Wh*Ww, Wh*Ww)
        softmax(attn, 3)
    }

    /// Get the learnable logit scale.
    ///
    /// ## Returns
    ///
    /// - Output tensor of shape (num_heads, 1, 1).
    fn logit_scale(&self) -> Tensor<B, 3> {
        // TODO(crutcher): I suspect this is a bug in the original code.
        // I *think* the authors thought this was log_10; and not log_e;
        // it doesn't make sense to use log_e with a scale of 10.0 here.
        self.logit_scale.val().clamp_max((1.0f64 / 0.01).ln()).exp()
    }

    /// Get the learnable relative position bias.
    ///
    /// ## Returns
    ///
    /// - Output tensor of shape (num_heads, Wh*Ww, Wh*Ww).
    fn relative_pos_bias(&self) -> Tensor<B, 3> {
        self.rpb_module.forward()
    }

    /// Encode the attention logits with the logit scale and relative position bias.
    ///
    /// ## Arguments
    ///
    /// - `attn`: Attention logits tensor of shape (b_nw, num_heads, Wh*Ww, Wh*Ww).
    ///
    /// ## Returns
    ///
    /// - Output tensor of shape (b_nw, num_heads, Wh*Ww, Wh*Ww).
    fn encode_attention(
        &self,
        attn: Tensor<B, 4>,
    ) -> Tensor<B, 4> {
        attn * self.logit_scale().unsqueeze() + self.relative_pos_bias().unsqueeze()
    }
}

impl WindowAttentionConfig {
    pub fn init<B: Backend>(
        &self,
        device: &B::Device,
    ) -> WindowAttention<B> {
        let d_input = self.d_input();
        let num_heads = self.num_heads();
        let window_size = self.window_shape();

        WindowAttention {
            d_input,
            num_heads,
            q_linear: LinearConfig::new(d_input, d_input)
                .with_bias(self.enable_qkv_bias)
                .init(device),
            k_linear: LinearConfig::new(d_input, d_input).init(device),
            v_linear: LinearConfig::new(d_input, d_input)
                .with_bias(self.enable_qkv_bias)
                .init(device),
            logit_scale: Param::initialized(
                ParamId::new(),
                // TODO(crutcher): I suspect this is a bug in the original code.
                // I *think* the authors thought this was log_10; and not log_e;
                // it doesn't make sense to use log_e with a scale of 10.0 here.
                Tensor::<B, 3>::ones([num_heads, 1, 1], device)
                    .mul_scalar(10.0)
                    .log(),
            ),
            attn_drop: DropoutConfig {
                prob: self.attn_drop,
            }
            .init(),
            rpb_module: RelativePositionBiasConfig::new(num_heads, window_size)
                .init_offset_grid_rpb(device),
            proj: LinearConfig::new(d_input, d_input)
                .with_bias(false)
                .init(device),
            proj_drop: DropoutConfig {
                prob: self.proj_drop,
            }
            .init(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::Distribution;
    use burn_contracts::assert_tensor;

    #[test]
    fn test_wa() {
        let b = 3;
        let num_windows = 2;

        let window_size = 4;

        let num_heads = 5;
        let cph = 3;
        let channels = num_heads * cph;

        let config = WindowAttentionConfig::new(channels, [window_size, window_size], num_heads);

        let device = Default::default();
        let attn_mod = config.init::<NdArray>(&device);

        assert_eq!(attn_mod.d_input(), channels);
        assert_eq!(attn_mod.window_shape(), [window_size, window_size]);
        assert_eq!(attn_mod.num_heads(), num_heads);

        let distribution = Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 3>::random(
            [b * num_windows, window_size * window_size, channels],
            distribution,
            &device,
        );

        let res = attn_mod.forward(input, None);
        assert_tensor(&res).has_dims([b * num_windows, window_size * window_size, channels]);
    }
}
