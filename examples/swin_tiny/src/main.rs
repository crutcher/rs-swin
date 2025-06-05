mod data;

use crate::data::{CinicBatch, CinicBatcher, CinicDataset};
use bimm_swin::models::swin::v2::transformer::{
    LayerConfig, SwinTransformerV2, SwinTransformerV2Config,
};
use burn::backend::{Autodiff, Cuda};
use burn::config::Config;
use burn::data::dataloader::DataLoaderBuilder;
use burn::grad_clipping::GradientClippingConfig;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::prelude::{Backend, Int, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::{ClassificationOutput, LearnerBuilder};
use burn::train::{TrainOutput, TrainStep, ValidStep};
use rand::{rng};
use rand::seq::SliceRandom;
use rs_cinic_10_index::Cinic10Index;
use rs_cinic_10_index::index::ObjectClass;
use strum::EnumCount;

#[derive(Config, Debug)]
pub struct ModelConfig {
    pub swin: SwinTransformerV2Config,
}

impl ModelConfig {
    pub fn init<B: Backend>(
        self,
        device: &B::Device,
    ) -> Model<B> {
        Model {
            swin: self.swin.init::<B>(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct Model<B: Backend> {
    pub swin: SwinTransformerV2<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward_classification(
        &self,
        images: Tensor<B, 4>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.swin.forward(images);

        let loss = CrossEntropyLossConfig::new()
            .with_smoothing(Some(0.1))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<CinicBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(
        &self,
        batch: CinicBatch<B>,
    ) -> TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);

        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<CinicBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(
        &self,
        batch: CinicBatch<B>,
    ) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
pub struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: AdamWConfig,

    #[config(default = 10)]
    pub num_epochs: usize,

    #[config(default = 64)]
    pub batch_size: usize,

    #[config(default = 2)]
    pub num_workers: usize,

    #[config(default = 42)]
    pub seed: u64,

    #[config(default = 1.0e-5)]
    pub learning_rate: f64,
}

fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: &B::Device,
) {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let index: Cinic10Index = Default::default();

    let batcher = CinicBatcher::default();

    let dataloader_train = {
        let mut items = index.train.items.clone();
        items.extend(index.test.items.clone());

        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(CinicDataset {
                items
            })
    };

    let dataloader_test = {
        // TODO: something less dumb, using PartialDataset and ShuffledDataset, or a wrapper.
            let mut rg = rng();

            let mut items = index.valid.items.clone();
            items.shuffle(&mut rg);

            let take = items.len() / 10;

            let items = items.into_iter().take(take).collect::<Vec<_>>();

        DataLoaderBuilder::new(batcher.clone())
            .batch_size(config.batch_size * 2)
            .shuffle(config.seed)
            .num_workers(config.num_workers)
            .build(CinicDataset { items})
    };

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(device),
            config.optimizer.init(),
            config.learning_rate,
        );

    let model_trained = learner.fit(dataloader_train, dataloader_test);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");
}

fn main() {
    type B = Autodiff<Cuda>;

    let h = 32;
    let w = 32;
    let img_res = [h, w];
    let patch_size = 2;
    let channels = 3;
    let window_size = 4;
    let embed_dim = 2 * channels * patch_size * patch_size;
    let num_classes = ObjectClass::COUNT;

    let config = SwinTransformerV2Config::new(
        img_res,
        patch_size,
        channels,
        num_classes,
        embed_dim,
        vec![
            LayerConfig::new(2, 3),
            LayerConfig::new(6, 6),
            LayerConfig::new(2, 12),
        ],
    )
    .with_window_size(window_size)
    .with_drop_rate(0.2);

    let training_config = TrainingConfig::new(
        ModelConfig { swin: config },
        AdamWConfig::new()
            .with_grad_clipping(Some(GradientClippingConfig::Norm(1.0)))
    )
    .with_num_epochs(90)
    .with_batch_size(128)
    .with_num_workers(1);

    let device = Default::default();

    train::<B>("/tmp/swin_tiny_cinic10", training_config, &device);
}
