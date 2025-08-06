use bimm_firehose::core::schema::{ColumnSchema, FirehoseTableSchema};
use bimm_firehose::ops::image::ImageShape;
use bimm_firehose::ops::image::loader::{ColorType, ImageLoader, ResizeSpec};
use bimm_firehose::ops::image::tensor_loader::{ImageDimLayout, ImgToTensorConfig, TargetDType};
use burn::backend::Cuda;
use itertools::Itertools;
use rs_cinic_10_index::Cinic10Index;
use std::sync::Arc;

use bimm_firehose::core::operations::executor::FirehoseBatchExecutor;
use bimm_firehose::core::{FirehoseRowBatch, FirehoseRowReader, FirehoseRowWriter, ValueBox};
use bimm_firehose::ops::image::aug::{FlipSpec, ImageAugmenter};
use bimm_firehose::ops::init_burn_device_operator_environment;
use burn::prelude::Tensor;
use clap::Parser;
use std::time::Instant;

#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// Batch size for processing
    #[arg(short, long, default_value_t = 512)]
    batch_size: usize,

    /// Whether to simulate the stack operation
    #[arg(long, default_value_t = false)]
    simulate_stack: bool,
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let batch_size = args.batch_size;

    let index: Cinic10Index = Default::default();

    type B = Cuda;

    let device = Default::default();

    let env = Arc::new(init_burn_device_operator_environment::<B>(&device));

    // Define a processing schema, from `path` -> `image` -> `tensor`.
    let schema = {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<String>("path").with_description("path to the image"),
            ColumnSchema::new::<u64>("seed").with_description("aug seed"),
        ]);

        // Load the image from the path, resize it to 32x32 pixels, and convert it to RGB8.
        ImageLoader::default()
            .with_resize(ResizeSpec::new(ImageShape {
                width: 32,
                height: 32,
            }))
            .with_recolor(ColorType::Rgb8)
            .to_plan("path", "image")
            .apply_to_schema(&mut schema, env.as_ref())?;

        ImageAugmenter::new()
            .with_flip(FlipSpec::new().with_vertical(0.5).with_horizontal(0.5))
            .with_rotate(true)
            .to_plan("seed", "image", "aug")
            .apply_to_schema(&mut schema, env.as_ref())?;

        // Convert the image to a tensor of shape (32, 32, 3) with float32 dtype.
        ImgToTensorConfig::new()
            .with_dtype(TargetDType::F32)
            .with_dim_layout(ImageDimLayout::HWC)
            .to_plan("aug", "tensor")
            .apply_to_schema(&mut schema, env.as_ref())?;

        Arc::new(schema)
    };

    let executor = Arc::new(
        bimm_firehose::core::operations::executor::SequentialBatchExecutor::new(
            schema.clone(),
            env.clone(),
        )?,
    );

    // Track the time it takes to process the dataset in batches.
    let mut durations = Vec::new();

    // Simulate processing the dataset in batches, without threading.
    for chunk in (0..index.test.len()).chunks(batch_size).into_iter() {
        let start_time = Instant::now();

        let selection = chunk.collect::<Vec<_>>();

        // Fill a batch with the selected paths.
        let mut batch = FirehoseRowBatch::new(schema.clone());
        for &idx in selection.iter() {
            let item = &index.test.items[idx];

            let row = batch.new_row();
            row.set("seed", ValueBox::serializing::<u64>(idx as u64)?);
            row.set(
                "path",
                ValueBox::serializing(item.path.to_string_lossy().to_string())?,
            );
        }

        // Run the batch.
        executor.execute_batch(&mut batch)?;

        if args.simulate_stack {
            // Simulate the batch collation function.
            let _stack: Tensor<B, 4> = Tensor::stack(
                batch
                    .iter()
                    .map(|row| {
                        row.get("tensor")
                            .unwrap()
                            .as_ref::<Tensor<B, 3>>()
                            .unwrap()
                            .clone()
                    })
                    .collect::<Vec<_>>(),
                0,
            );
        }

        let end_time = Instant::now();
        let duration = end_time.duration_since(start_time);
        durations.push(duration);
    }

    let total_duration: std::time::Duration = durations.iter().sum();
    let batch_duration = total_duration / durations.len() as u32;
    let item_duration = batch_duration / batch_size as u32;

    println!("Loaded {} images", index.test.len());
    println!("Total duration: {total_duration:?}");
    println!("batch_size: {batch_size}");
    println!("batch duration: {batch_duration:?}");
    println!("item duration: {item_duration:?}");

    Ok(())
}
