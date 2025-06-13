use burn::data::dataloader::Dataset;
use burn::data::dataloader::batcher::Batcher;
use burn::prelude::{Backend, Int, Tensor};
use enum_ordinalize::Ordinalize;
use rs_cinic_10_burn::load_hwc_u8_tensor_image;
use rs_cinic_10_index::index::DatasetItem;

pub struct CinicDataset {
    pub items: Vec<DatasetItem>,
}

impl Dataset<DatasetItem> for CinicDataset {
    fn get(
        &self,
        index: usize,
    ) -> Option<DatasetItem> {
        Some(self.items[index].clone())
    }

    fn len(&self) -> usize {
        self.items.len()
    }
}

#[derive(Clone, Debug, Default)]
pub struct CinicBatcher {}

#[derive(Clone, Debug)]
pub struct CinicBatch<B: Backend> {
    pub images: Tensor<B, 4>,       // Shape: [B, C, H, W]
    pub targets: Tensor<B, 1, Int>, // Shape: [B]
}

impl<B: Backend> Batcher<B, DatasetItem, CinicBatch<B>> for CinicBatcher {
    fn batch(
        &self,
        items: Vec<DatasetItem>,
        device: &B::Device,
    ) -> CinicBatch<B> {
        let images = items
            .iter()
            .map(|item| load_hwc_u8_tensor_image(&item.path, device).unwrap())
            .collect::<Vec<_>>();
        let images: Tensor<B, 4> = Tensor::stack(images, 0);
        // Change from [B, H, W, C] to [B, C, H, W]
        let images = images.permute([0, 3, 1, 2]);

        let images = images.div_scalar(255.0);

        // Fixed normalization for Cinic-10 dataset
        let images = images.sub_scalar(0.4);
        let images = images.div_scalar(0.2);

        let ordinals: Vec<i32> = items
            .iter()
            .map(|item| item.class.ordinal() as i32)
            .collect();

        let targets: Tensor<B, 1, Int> = Tensor::from_data(ordinals.as_slice(), device);

        CinicBatch { images, targets }
    }
}
