use burn::prelude::{Backend, Tensor};

/// Apply an attention mask.
///
/// ## Parameters
///
/// - `b_nw`: Batch size times number of windows.
/// - `n`: Number of elements in the input tensor, Wh*Ww.
/// - `num_heads`: Number of attention heads.
/// - `attn`: Attention logits tensor of shape (b_nw, num_heads, Wh*Ww, Wh*Ww).
/// - `mask`: Mask tensor of shape (num_windows, Wh*Ww, Wh*Ww).
///
/// ## Returns
///
/// - Output tensor of shape (b_nw, num_heads, Wh*Ww, Wh*Ww).
pub fn apply_attention_mask<B: Backend>(
    b_nw: usize,
    n: usize,
    num_heads: usize,
    attn: Tensor<B, 4>,
    mask: Tensor<B, 3>,
) -> Tensor<B, 4> {
    // Attention mask
    let num_windows = mask.dims()[0];
    let b = b_nw / num_windows;

    let attn = attn.reshape([b, num_windows, num_heads, n, n]);

    let mask = mask.unsqueeze_dim::<4>(1).unsqueeze::<5>();
    // 1, num_windows, 1, Wh*Ww, Wh*Ww

    let attn: Tensor<B, 5> = attn + mask;
    // b, num_windows, num_heads, Wh*Ww, Wh*Ww

    attn.reshape([-1, num_heads as i32, n as i32, n as i32])
    // b_nw, num_heads, Wh*Ww, Wh*Ww
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::{Tensor, s};
    use burn_contracts::assert_tensor;

    #[test]
    fn test_apply_attention_mask() {
        let b = 2;
        let nw = 2;
        let b_nw = b * nw;
        let ws = 2;
        let n = ws * ws;
        let num_heads = 5;

        let device = Default::default();
        let attn = Tensor::<NdArray, 4>::zeros([b_nw, num_heads, n, n], &device);
        // (b*nw, num_heads, ws*ws, ws*ws)

        // (nw, ws*ws, ws*ws)
        let mask = Tensor::<NdArray, 3>::from_data(
            [
                [
                    [0.0, 0.25, 0.5, 0.75],
                    [1.0, 1.25, 1.5, 1.75],
                    [2.0, 2.25, 2.5, 2.75],
                    [3.0, 3.25, 3.5, 3.75],
                ],
                [
                    [1.0, 0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0, 0.0],
                    [0.0, 0.0, 0.0, 1.0],
                ],
            ],
            &device,
        );

        let res = apply_attention_mask(b_nw, n, num_heads, attn, mask.clone());
        assert_tensor(&res).has_dims([b_nw, num_heads, n, n]);

        let res = res.reshape([b, nw, num_heads, n, n]);

        for bi in 0..b {
            for wi in 0..nw {
                let window = res
                    .clone()
                    .slice(s![bi, wi, .., ..])
                    .squeeze::<4>(0)
                    .squeeze::<3>(0);

                let wmask: Tensor<NdArray, 2> = mask.clone().slice(s![wi, .., ..]).squeeze::<2>(0);

                for hi in 0..num_heads {
                    let hattn = window.clone().slice(s![hi, .., ..]).squeeze::<2>(0);

                    hattn.to_data().assert_eq(&wmask.to_data(), true);
                }
            }
        }
    }
}
