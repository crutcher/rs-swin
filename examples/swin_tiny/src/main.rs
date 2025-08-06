#![recursion_limit = "256"]
extern crate core;

use bimm::models::swin::v2::transformer::{
    LayerConfig, SwinTransformerV2, SwinTransformerV2Config,
};
use bimm_firehose::burn::batcher::{
    BatcherInputAdapter, BatcherOutputAdapter, FirehoseExecutorBatcher,
};
use bimm_firehose::core::operations::executor::SequentialBatchExecutor;
use bimm_firehose::core::schema::ColumnSchema;
use bimm_firehose::core::{
    FirehoseRowBatch, FirehoseRowReader, FirehoseRowWriter, FirehoseTableSchema, ValueBox,
};
use bimm_firehose::ops::image::ImageShape;
use bimm_firehose::ops::image::aug::{ColorType, FlipSpec, ImageAugmenter};
use bimm_firehose::ops::image::loader::{ImageLoader, ResizeSpec};
use bimm_firehose::ops::image::tensor_loader::{ImageToTensorData, stack_tensor_data_column};
use bimm_firehose::ops::init_default_operator_environment;
use burn::backend::{Autodiff, Cuda};
use burn::config::Config;
use burn::data::dataloader::{DataLoaderBuilder, Dataset};
use burn::data::dataset::transform::{SamplerDataset, ShuffledDataset};
use burn::grad_clipping::GradientClippingConfig;
use burn::lr_scheduler::cosine::CosineAnnealingLrSchedulerConfig;
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::optim::AdamWConfig;
use burn::prelude::{Backend, Int, Tensor};
use burn::record::CompactRecorder;
use burn::tensor::backend::AutodiffBackend;
use burn::train::metric::store::{Aggregate, Direction, Split};
use burn::train::metric::{
    AccuracyMetric, CpuMemory, CpuTemperature, CpuUse, CudaMetric, LearningRateMetric, LossMetric,
    TopKAccuracyMetric,
};
use burn::train::{
    ClassificationOutput, LearnerBuilder, MetricEarlyStoppingStrategy, StoppingCondition,
};
use burn::train::{TrainOutput, TrainStep, ValidStep};
use enum_ordinalize::Ordinalize;
use rand::{Rng, rng};
use rs_cinic_10_index::Cinic10Index;
use rs_cinic_10_index::index::{DatasetItem, ObjectClass};
use std::sync::Arc;
use strum::EnumCount;

const PATH_COLUMN: &str = "path";
const SEED_COLUMN: &str = "seed";
const CLASS_COLUMN: &str = "class";
const IMAGE_COLUMN: &str = "image";
const AUG_COLUMN: &str = "aug";
const DATA_COLUMN: &str = "data";

fn main() -> anyhow::Result<()> {
    type B = Autodiff<Cuda>;

    let h = 32;
    let w = 32;
    let channels = 3;

    let img_res = [h, w];
    let patch_size = 2;
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
            LayerConfig::new(8, 6),
            LayerConfig::new(8, 8),
            LayerConfig::new(6, 12),
        ],
    )
    .with_window_size(window_size)
    .with_attn_drop_rate(0.1)
    .with_drop_rate(0.1);

    let training_config = TrainingConfig::new(
        ModelConfig { swin: config },
        AdamWConfig::new()
            .with_weight_decay(0.05)
            .with_grad_clipping(Some(GradientClippingConfig::Norm(5.0))),
    )
    .with_batch_size(500)
    .with_learning_rate(1.0e-3)
    .with_min_learning_rate(1.0e-5)
    .with_num_epochs(60)
    .with_num_workers(Some(1));

    let device = Default::default();

    train::<B>("/tmp/swin_tiny_cinic10", training_config, &device)
}

/// Config for training the model.
#[derive(Config)]
pub struct TrainingConfig {
    /// The inner model config.
    pub model: ModelConfig,

    /// The optimizer config.
    pub optimizer: AdamWConfig,

    /// Ratio of oversampling the training dataset.
    #[config(default = "2.5")]
    pub oversample_ratio: f64,

    /// Number of epochs to train the model.
    #[config(default = 10)]
    pub num_epochs: usize,

    /// Batch size for training and validation.
    #[config(default = 64)]
    pub batch_size: usize,

    /// Number of workers for data loading.
    #[config(default = "Option::None")]
    pub num_workers: Option<usize>,

    /// Random seed for reproducibility.
    #[config(default = 42)]
    pub seed: u64,

    /// Learning rate for the optimizer.
    #[config(default = 1.0e-5)]
    pub learning_rate: f64,

    /// Learning rate for the optimizer.
    #[config(default = 1.0e-7)]
    pub min_learning_rate: f64,
}

/// Create the artifact directory for saving training artifacts.
fn create_artifact_dir(artifact_dir: &str) {
    // Remove existing artifacts before to get an accurate learner summary
    std::fs::remove_dir_all(artifact_dir).ok();
    std::fs::create_dir_all(artifact_dir).ok();
}

/// Train the model with the given configuration and devices.
pub fn train<B: AutodiffBackend>(
    artifact_dir: &str,
    config: TrainingConfig,
    device: &B::Device,
) -> anyhow::Result<()> {
    create_artifact_dir(artifact_dir);

    config
        .save(format!("{artifact_dir}/config.json"))
        .expect("Config should be saved successfully");

    B::seed(config.seed);

    let cinic10_index: Cinic10Index = Default::default();

    let firehose_env = Arc::new(init_default_operator_environment());

    let common_schema = {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<String>(PATH_COLUMN).with_description("path to the image"),
            ColumnSchema::new::<i32>(CLASS_COLUMN).with_description("image class"),
            ColumnSchema::new::<u64>(SEED_COLUMN).with_description("instance rng seed"),
        ]);

        // Load the image from the path, resize it to 32x32 pixels, and convert it to RGB8.
        ImageLoader::default()
            .with_resize(ResizeSpec::new(ImageShape {
                width: 32,
                height: 32,
            }))
            .with_recolor(ColorType::Rgb8)
            .to_plan(PATH_COLUMN, IMAGE_COLUMN)
            .apply_to_schema(&mut schema, firehose_env.as_ref())?;

        schema
    };

    let train_dataloader = {
        let ds = CinicDataset {
            items: cinic10_index.train.items.clone(),
        };
        let ds = ShuffledDataset::with_seed(ds, config.seed);
        let num_samples = (config.oversample_ratio * (ds.len() as f64)).ceil() as usize;
        let ds = SamplerDataset::without_replacement(ds, num_samples);

        let schema = Arc::new({
            let mut schema = common_schema.clone();

            ImageAugmenter::new()
                .with_flip(FlipSpec::new().with_horizontal(0.5))
                .to_plan(SEED_COLUMN, IMAGE_COLUMN, AUG_COLUMN)
                .apply_to_schema(&mut schema, firehose_env.as_ref())?;

            // Convert the image to a tensor of shape (3, 32, 32) with float32 dtype.
            ImageToTensorData::new()
                .to_plan(AUG_COLUMN, DATA_COLUMN)
                .apply_to_schema(&mut schema, firehose_env.as_ref())?;

            schema
        });

        let batcher = FirehoseExecutorBatcher::new(
            Arc::new(SequentialBatchExecutor::new(
                schema.clone(),
                firehose_env.clone(),
            )?),
            Arc::new(InputAdapter::new(schema.clone())),
            Arc::new(OutputAdapter::<B>::default()),
        );

        let mut builder = DataLoaderBuilder::new(batcher).batch_size(config.batch_size);
        if let Some(num_workers) = config.num_workers {
            builder = builder.num_workers(num_workers);
        }
        builder.build(ds)
    };

    let validation_dataloader = {
        let ds = CinicDataset {
            items: cinic10_index.valid.items.clone(),
        };

        let schema = Arc::new({
            let mut schema = common_schema.clone();

            // Convert the image to a tensor of shape (3, 32, 32) with float32 dtype.
            ImageToTensorData::new()
                .to_plan(IMAGE_COLUMN, DATA_COLUMN)
                .apply_to_schema(&mut schema, firehose_env.as_ref())?;

            schema
        });

        let batcher = FirehoseExecutorBatcher::new(
            Arc::new(SequentialBatchExecutor::new(
                schema.clone(),
                firehose_env.clone(),
            )?),
            Arc::new(InputAdapter::new(schema.clone())),
            // Use the InnerBackend for validation.
            Arc::new(OutputAdapter::<B::InnerBackend>::default()),
        );

        let mut builder = DataLoaderBuilder::new(batcher).batch_size(config.batch_size);
        if let Some(num_workers) = config.num_workers {
            builder = builder.num_workers(num_workers);
        }
        builder.build(ds)
    };

    let num_batches = train_dataloader.num_items() / config.batch_size;

    let lr_scheduler = CosineAnnealingLrSchedulerConfig::new(
        config.learning_rate,
        num_batches * config.num_epochs,
    )
    .with_min_lr(config.min_learning_rate)
    .init()
    .map_err(|e| anyhow::anyhow!("Failed to initialize learning rate scheduler: {}", e))?;

    let learner = LearnerBuilder::new(artifact_dir)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(TopKAccuracyMetric::new(2))
        .metric_valid_numeric(TopKAccuracyMetric::new(2))
        .metric_train(CudaMetric::new())
        .metric_valid(CudaMetric::new())
        .metric_train_numeric(CpuUse::new())
        .metric_valid_numeric(CpuUse::new())
        .metric_train_numeric(CpuMemory::new())
        .metric_valid_numeric(CpuMemory::new())
        .metric_train_numeric(CpuTemperature::new())
        .metric_valid_numeric(CpuTemperature::new())
        .metric_train_numeric(LearningRateMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .early_stopping(MetricEarlyStoppingStrategy::new(
            &LossMetric::<B>::new(),
            Aggregate::Mean,
            Direction::Lowest,
            Split::Valid,
            StoppingCondition::NoImprovementSince { n_epochs: 6 },
        ))
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .summary()
        .build(
            config.model.init::<B>(device),
            config.optimizer.init(),
            lr_scheduler,
        );

    let model_trained = learner.fit(train_dataloader, validation_dataloader);

    model_trained
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .expect("Trained model should be saved successfully");

    Ok(())
}

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
            // .with_smoothing(Some(0.1))
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<(Tensor<B, 4>, Tensor<B, 1, Int>), ClassificationOutput<B>>
    for Model<B>
{
    fn step(
        &self,
        batch: (Tensor<B, 4>, Tensor<B, 1, Int>),
    ) -> TrainOutput<ClassificationOutput<B>> {
        let (images, targets) = batch;
        let item = self.forward_classification(images, targets);
        TrainOutput::new(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<(Tensor<B, 4>, Tensor<B, 1, Int>), ClassificationOutput<B>>
    for Model<B>
{
    fn step(
        &self,
        batch: (Tensor<B, 4>, Tensor<B, 1, Int>),
    ) -> ClassificationOutput<B> {
        let (images, targets) = batch;
        self.forward_classification(images, targets)
    }
}

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

fn init_batch_from_dataset_items(
    inputs: &Vec<DatasetItem>,
    batch: &mut FirehoseRowBatch,
) -> anyhow::Result<()> {
    let mut local_rng = rng();
    for item in inputs {
        let row = batch.new_row();
        row.set(
            PATH_COLUMN,
            ValueBox::serializing(item.path.to_string_lossy().to_string())?,
        );
        row.set(
            CLASS_COLUMN,
            ValueBox::serializing(item.class.ordinal() as i32)?,
        );
        row.set(
            SEED_COLUMN,
            ValueBox::serializing(local_rng.random::<u64>())?,
        );
    }

    Ok(())
}

struct InputAdapter {
    schema: Arc<FirehoseTableSchema>,
}
impl InputAdapter {
    pub fn new(schema: Arc<FirehoseTableSchema>) -> Self {
        Self { schema }
    }
}
impl BatcherInputAdapter<DatasetItem> for InputAdapter {
    fn apply(
        &self,
        inputs: Vec<DatasetItem>,
    ) -> anyhow::Result<FirehoseRowBatch> {
        let mut batch = FirehoseRowBatch::new(self.schema.clone());
        init_batch_from_dataset_items(&inputs, &mut batch)?;
        Ok(batch)
    }
}

struct OutputAdapter<B: Backend> {
    phantom: std::marker::PhantomData<B>,
}
impl<B> Default for OutputAdapter<B>
where
    B: Backend,
{
    fn default() -> Self {
        Self {
            phantom: std::marker::PhantomData,
        }
    }
}
impl<B: Backend> BatcherOutputAdapter<B, (Tensor<B, 4>, Tensor<B, 1, Int>)> for OutputAdapter<B> {
    fn apply(
        &self,
        batch: &FirehoseRowBatch,
        device: &B::Device,
    ) -> anyhow::Result<(Tensor<B, 4>, Tensor<B, 1, Int>)> {
        let image_batch = Tensor::<B, 4>::from_data(
            stack_tensor_data_column(batch, DATA_COLUMN)
                .expect("Failed to stack tensor data column"),
            device,
        )
        // Change from [B, H, W, C] to [B, C, H, W]
        .permute([0, 3, 1, 2])
        // Fixed normalization for Cinic-10 dataset
        .sub_scalar(0.4)
        // Fixed normalization for Cinic-10 dataset
        .div_scalar(0.2);

        let target_batch = Tensor::from_data(
            batch
                .iter()
                .map(|row| row.get(CLASS_COLUMN).unwrap().parse_as::<i32>().unwrap())
                .collect::<Vec<_>>()
                .as_slice(),
            device,
        );

        Ok((image_batch, target_batch))
    }
}
