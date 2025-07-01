use bimm_contracts::{ShapeContract, shape_contract};
use burn::prelude::{Backend, Tensor};
use burn::tensor::BasicOps;

/// Window Partition
///
/// ## Parameters
///
/// - `tensor`: Input tensor of shape (B, H, W, C).
/// - `window_size`: Window size.
///
/// ## Returns
///   - Output tensor of shape (B * h_windows * w_windows, window_size, window_size, C).
#[inline]
#[must_use]
pub fn window_partition<B: Backend, K>(
    tensor: Tensor<B, 4, K>,
    window_size: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    static CONTRACT: ShapeContract = shape_contract!(
        "batch",
        "h_wins" * "window_size",
        "w_wins" * "window_size",
        "channels"
    );
    let [b, h_wins, w_wins, c] = CONTRACT.unpack_shape(
        &tensor,
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)],
    );

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
#[inline]
#[must_use]
pub fn window_reverse<B: Backend, K>(
    windows: Tensor<B, 4, K>,
    window_size: usize,
    h: usize,
    w: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    static CONTRACT: ShapeContract = shape_contract!(
        "batch" * "h_wins" * "w_wins",
        "window_size",
        "window_size",
        "channels"
    );

    let h_wins = h / window_size;
    let w_wins = w / window_size;

    let [b, c] = CONTRACT.unpack_shape(
        &windows,
        &["batch", "channels"],
        &[
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("window_size", window_size),
        ],
    );

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
    use burn::tensor::{Distribution, Tolerance};

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

        let windows = window_partition(input.clone(), window_size);

        assert_eq!(
            &windows.dims(),
            &[b * h_wins * w_wins, window_size, window_size, channels]
        );

        let reverse = window_reverse(windows, window_size, h, w);
        assert_eq!(&reverse.dims(), &[b, h, w, channels]);

        reverse
            .to_data()
            .assert_approx_eq(&input.to_data(), Tolerance::<f64>::default());
    }
}
