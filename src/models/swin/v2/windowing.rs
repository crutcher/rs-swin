use burn::prelude::{Backend, Tensor};
use burn_contracts::assert_tensor;

/// Window Partition
///
/// ## Parameters
///
/// - `tensor`: Input tensor of shape (B, H, W, C).
/// - `window_size`: Window size.
///
/// ## Returns
///   - Output tensor of shape (B * h_windows * w_windows, window_size, window_size, C).
pub fn window_partition<B: Backend>(
    tensor: Tensor<B, 4>,
    window_size: usize,
) -> Tensor<B, 4> {
    let [b, h_wins, w_wins, c] = assert_tensor(&tensor)
        .unpacks_shape(
            ["b", "h_wins", "w_wins", "c"],
            "b (h_wins window_size) (w_wins window_size) c",
            &[("window_size", window_size)],
        )
        .unwrap();

    tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c])
}

/// Window Reverse
///
/// ## Parameters
/// - `windows`: Input tensor of shape (B * h_windows * w_windows, window_size, window_size, C).
/// - `window_size`: Window size.
/// - `h`: Height of the original image.
/// - `w`: Width of the original image.
///
/// ## Returns
/// - Output tensor of shape (B, H, W, C).
pub fn window_reverse<B: Backend>(
    windows: Tensor<B, 4>,
    window_size: usize,
    h: usize,
    w: usize,
) -> Tensor<B, 4> {
    let h_wins = h / window_size;
    let w_wins = w / window_size;

    let [b, c] = assert_tensor(&windows)
        .unpacks_shape(
            ["b", "c"],
            "(b h_wins w_wins) window_size window_size c",
            &[
                ("h_wins", h_wins),
                ("w_wins", w_wins),
                ("window_size", window_size),
            ],
        )
        .unwrap();

    windows
        .reshape([b, h_wins, w_wins, window_size, window_size, c])
        .swap_dims(2, 3)
        .reshape([b, h, w, c])
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::backend::NdArray;
    use burn::prelude::Tensor;
    use burn::tensor::Distribution;
    use burn_contracts::assert_tensor;

    #[test]
    fn test_window_partition() {
        let device = Default::default();

        let b = 3;
        let window_size = 4;
        let channels = 3;

        let h_wins = 2;
        let w_wins = 3;
        let h = h_wins * window_size;
        let w = w_wins * window_size;

        let distribution = Distribution::Uniform(0.0, 1.0);
        let input = Tensor::<NdArray, 4>::random([b, h, w, channels], distribution, &device);

        let windows = window_partition::<NdArray>(input.clone(), window_size);

        assert_tensor(&windows).has_dims([b * h_wins * w_wins, window_size, window_size, channels]);

        let reverse = window_reverse::<NdArray>(windows, window_size, h, w);

        assert_tensor(&reverse)
            .has_dims([b, h, w, channels])
            .is_close(&input, None, None);
    }
}
