//! SW-MSA
//!
//! Shifted Window Multi-Head Self-Attention
//!
//! See: https://arxiv.org/pdf/2103.14030
use crate::models::swin::v2::windowing::window_partition;
use burn::prelude::{Backend, Bool, Int, Tensor};

/// Internal function to create a shifted window image mask.
///
/// This function generates a mask for the shifted window attention mechanism.
///
/// ## Parameters
///
/// - `input_shape`: The shape of the input tensor.
/// - `window_size`: The size of the window.
/// - `shift_size`: The size of the shift.
/// - `device`: The device on which the tensor will be created.
///
/// ## Returns
///
/// A tensor representing the shifted window image mask.
fn sw_img_mask<B: Backend>(
    input_shape: [usize; 2],
    window_size: usize,
    shift_size: usize,
    device: &B::Device,
) -> Tensor<B, 2, Int> {
    let [h, w] = input_shape;

    let mut img_mask = Tensor::<B, 2, Int>::zeros([h, w], device);

    let h = h as i32;
    let w = w as i32;

    let window_size = window_size as i32;
    let shift_size = shift_size as i32;

    let h_slices = [
        0..(h - window_size) as usize,
        (h - window_size) as usize..(h - shift_size) as usize,
        (h - shift_size) as usize..h as usize,
    ];
    let w_slices = [
        0..(w - window_size) as usize,
        (w - window_size) as usize..(w - shift_size) as usize,
        (w - shift_size) as usize..w as usize,
    ];

    let mut cnt = 0;
    for h in h_slices.iter() {
        for w in w_slices.iter() {
            let slice_shape = img_mask.clone().slice([h.clone(), w.clone()]).dims();
            let val: Tensor<B, 1, Int> = Tensor::from_data([cnt], device);
            let val = val.unsqueeze::<2>().expand(slice_shape);

            img_mask = img_mask.slice_assign([h.clone(), w.clone()], val);
            cnt += 1;
        }
    }

    img_mask
}

/// Create a shifted window attention mask.
///
/// This function generates a mask for the shifted window attention mechanism.
///
/// ## Parameters
///
/// - `input_shape`: The shape of the input tensor.
/// - `window_size`: The size of the window.
/// - `shift_size`: The size of the shift.
/// - `device`: The device on which the tensor will be created.
///
/// ## Returns
///
/// A tensor representing the shifted window attention mask.
pub fn sw_attn_mask<B: Backend>(
    input_shape: [usize; 2],
    window_size: usize,
    shift_size: usize,
    device: &B::Device,
) -> Tensor<B, 3, Bool> {
    let img_mask = sw_img_mask(input_shape, window_size, shift_size, device);
    // ws, ws
    let img_mask = img_mask.unsqueeze_dim::<3>(2).unsqueeze::<4>();
    // b_nw=1, ws, ws, 1

    let mask_windows = window_partition(img_mask, window_size);
    // b_nw=1, ws, ws, 1

    let mask_windows = mask_windows.reshape([-1, (window_size * window_size) as i32]);
    // b_nw=nW, ws * ws

    let mask =
        mask_windows.clone().unsqueeze_dim::<3>(1) - mask_windows.clone().unsqueeze_dim::<3>(2);

    mask.not_equal_elem(0)
}

#[cfg(test)]
mod tests {
    use super::*;

    use burn::backend::NdArray;
    use burn::prelude::TensorData;

    #[test]
    fn test_attn_mask() {
        // let b_nw = 1;
        let device = Default::default();

        sw_attn_mask::<NdArray>([4, 4], 2, 1, &device)
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [
                        [false, false, false, false],
                        [false, false, false, false],
                        [false, false, false, false],
                        [false, false, false, false],
                    ],
                    [
                        [false, true, false, true],
                        [true, false, true, false],
                        [false, true, false, true],
                        [true, false, true, false],
                    ],
                    [
                        [false, false, true, true],
                        [false, false, true, true],
                        [true, true, false, false],
                        [true, true, false, false],
                    ],
                    [
                        [false, true, true, true],
                        [true, false, true, true],
                        [true, true, false, true],
                        [true, true, true, false],
                    ],
                ]),
                true,
            );

        sw_attn_mask::<NdArray>([6, 6], 3, 1, &device)
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                    ],
                    [
                        [false, false, true, false, false, true, false, false, true],
                        [false, false, true, false, false, true, false, false, true],
                        [true, true, false, true, true, false, true, true, false],
                        [false, false, true, false, false, true, false, false, true],
                        [false, false, true, false, false, true, false, false, true],
                        [true, true, false, true, true, false, true, true, false],
                        [false, false, true, false, false, true, false, false, true],
                        [false, false, true, false, false, true, false, false, true],
                        [true, true, false, true, true, false, true, true, false],
                    ],
                    [
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [false, false, false, false, false, false, true, true, true],
                        [true, true, true, true, true, true, false, false, false],
                        [true, true, true, true, true, true, false, false, false],
                        [true, true, true, true, true, true, false, false, false],
                    ],
                    [
                        [false, false, true, false, false, true, true, true, true],
                        [false, false, true, false, false, true, true, true, true],
                        [true, true, false, true, true, false, true, true, true],
                        [false, false, true, false, false, true, true, true, true],
                        [false, false, true, false, false, true, true, true, true],
                        [true, true, false, true, true, false, true, true, true],
                        [true, true, true, true, true, true, false, false, true],
                        [true, true, true, true, true, true, false, false, true],
                        [true, true, true, true, true, true, true, true, false],
                    ],
                ]),
                true,
            );

        sw_attn_mask::<NdArray>([6, 6], 3, 2, &device)
            .to_data()
            .assert_eq(
                &TensorData::from([
                    [
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                        [
                            false, false, false, false, false, false, false, false, false,
                        ],
                    ],
                    [
                        [false, true, true, false, true, true, false, true, true],
                        [true, false, false, true, false, false, true, false, false],
                        [true, false, false, true, false, false, true, false, false],
                        [false, true, true, false, true, true, false, true, true],
                        [true, false, false, true, false, false, true, false, false],
                        [true, false, false, true, false, false, true, false, false],
                        [false, true, true, false, true, true, false, true, true],
                        [true, false, false, true, false, false, true, false, false],
                        [true, false, false, true, false, false, true, false, false],
                    ],
                    [
                        [false, false, false, true, true, true, true, true, true],
                        [false, false, false, true, true, true, true, true, true],
                        [false, false, false, true, true, true, true, true, true],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                        [true, true, true, false, false, false, false, false, false],
                    ],
                    [
                        [false, true, true, true, true, true, true, true, true],
                        [true, false, false, true, true, true, true, true, true],
                        [true, false, false, true, true, true, true, true, true],
                        [true, true, true, false, true, true, false, true, true],
                        [true, true, true, true, false, false, true, false, false],
                        [true, true, true, true, false, false, true, false, false],
                        [true, true, true, false, true, true, false, true, true],
                        [true, true, true, true, false, false, true, false, false],
                        [true, true, true, true, false, false, true, false, false],
                    ],
                ]),
                true,
            );
    }
}
