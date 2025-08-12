use crate::core::operations::operator::FirehoseOperator;
use crate::core::operations::planner::OperationPlan;
use crate::core::rows::FirehoseRowTransaction;
use crate::core::{FirehoseRowReader, FirehoseRowWriter};
use crate::ops;
use crate::ops::image::burn::{IMAGE_TO_TENSOR_DATA, pixeldepth_support};
use burn::data::dataset::vision::PixelDepth;
use burn::prelude::TensorData;
use image::{ColorType, DynamicImage};
use serde::{Deserialize, Serialize};

/// The `ImageToTensorData` operator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImageToTensorData {}

impl Default for ImageToTensorData {
    fn default() -> Self {
        Self::new()
    }
}

impl ImageToTensorData {
    /// Creates a new `ImgToTensorConfig`.
    pub fn new() -> Self {
        ImageToTensorData {}
    }

    /// Converts this configuration to an `OperationPlanner` for the `ImgToTensor` operator.
    ///
    /// # Arguments
    ///
    /// * `image_column` - The name of the input column containing the image.
    /// * `data_column` - The name of the output column for the tensor data.
    pub fn to_plan(
        self,
        image_column: &str,
        data_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(IMAGE_TO_TENSOR_DATA)
            .with_input("image", image_column)
            .with_output("data", data_column)
            .with_config(self)
    }
}

impl FirehoseOperator for ImageToTensorData {
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        let image: &DynamicImage = txn.expect_get_ref("image");

        let height = image.height() as usize;
        let width = image.width() as usize;
        let colors = image.color().channel_count() as usize;
        let shape = vec![height, width, colors];

        let pixvec = ops::image::burn::image_to_pixvec(image);
        let data: Vec<f32> = pixvec
            .iter()
            .map(|p| pixeldepth_support::pixel_depth_to_f32(p.clone()))
            .collect();

        let data = TensorData::new(data, shape);

        txn.expect_set_boxing("data", data);

        Ok(())
    }
}

/// Converts an image to a vector of pixel depths.
pub fn image_to_pixvec(image: &DynamicImage) -> Vec<PixelDepth> {
    let image = image.clone();
    // Image as Vec<PixelDepth>
    match image.color() {
        ColorType::L8 => image
            .into_luma8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::La8 => image
            .into_luma_alpha8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::L16 => image
            .into_luma16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::La16 => image
            .into_luma_alpha16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::Rgb8 => image
            .into_rgb8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::Rgba8 => image
            .into_rgba8()
            .iter()
            .map(|&x| PixelDepth::U8(x))
            .collect(),
        ColorType::Rgb16 => image
            .into_rgb16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::Rgba16 => image
            .into_rgba16()
            .iter()
            .map(|&x| PixelDepth::U16(x))
            .collect(),
        ColorType::Rgb32F => image
            .into_rgb32f()
            .iter()
            .map(|&x| PixelDepth::F32(x))
            .collect(),
        ColorType::Rgba32F => image
            .into_rgba32f()
            .iter()
            .map(|&x| PixelDepth::F32(x))
            .collect(),
        _ => panic!("Unrecognized image color type"),
    }
}
