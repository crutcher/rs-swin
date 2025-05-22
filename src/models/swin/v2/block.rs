use crate::layers::drop::DropPath;
use crate::models::swin::v2::attention::WindowAttention;
use crate::models::swin::v2::mlp::Mlp;
use burn::prelude::{Backend, Tensor};

pub struct TransformerBlock<B: Backend> {
    // norm1
    // norm2
    pub attn: WindowAttention<B>,
    pub drop_path: DropPath,
    pub mlp: Mlp<B>,

    // nw, ws, ws, 1
    // nw, ws * ws
    // nw, 1, ws * ws
    // nw, 1, 1, ws * ws
    pub attn_mask: Option<Tensor<B, 4>>,
}
