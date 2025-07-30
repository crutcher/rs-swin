use bimm_firehose::core::operations::environment::new_default_operator_environment;
use bimm_firehose::core::schema::{ColumnSchema, FirehoseTableSchema};
use bimm_firehose::ops::image::ImageShape;
use bimm_firehose::ops::image::loader::{ColorType, ImageLoader, ResizeSpec};
use bimm_firehose::ops::image::tensor_loader::{
    ImgToTensorConfig, TargetDType, img_to_tensor_op_binding,
};
use burn::backend::Cuda;
use itertools::Itertools;
use rs_cinic_10_index::Cinic10Index;
use std::sync::Arc;

use bimm_firehose::core::operations::executor::{FirehoseBatchExecutor, SequentialBatchExecutor};
use bimm_firehose::core::rows::RowBatch;
use burn::prelude::Tensor;
use std::time::Instant;

fn main() -> anyhow::Result<()> {
    let index: Cinic10Index = Default::default();

    type B = Cuda;

    let device = Default::default();

    // Define a firehose environment, extended with a tensor load op.
    let env = {
        let mut env = new_default_operator_environment();
        env.add_binding(img_to_tensor_op_binding::<B>(&device))?;

        Arc::new(env)
    };

    let executor = SequentialBatchExecutor::new(env.clone());

    // Define a processing schema, from `path` -> `image` -> `tensor`.
    let schema = {
        let mut schema = FirehoseTableSchema::from_columns(&[
            ColumnSchema::new::<String>("path").with_description("path to the image")
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

        // Convert the image to a tensor of shape (32, 32, 3) with float32 dtype.
        ImgToTensorConfig::new()
            .with_dtype(TargetDType::F32)
            .to_plan("image", "tensor")
            .apply_to_schema(&mut schema, env.as_ref())?;

        schema
    };

    // Track the time it takes to process the dataset in batches.
    let mut durations = Vec::new();

    // Simulate processing the dataset in batches, without threading.
    let batch_size = 512;
    for chunk in (0..index.test.len()).chunks(batch_size).into_iter() {
        let start_time = Instant::now();

        let selection = chunk.collect::<Vec<_>>();

        // Fill a batch with the selected paths.
        let mut batch = RowBatch::with_size(schema.clone(), selection.len());
        for (i, &idx) in selection.iter().enumerate() {
            let item = &index.test.items[idx];
            batch[i].set_column(
                &schema,
                "path",
                Some(Arc::new(item.path.to_string_lossy().to_string())),
            );
        }

        // Run the batch.
        executor.execute_batch(&mut batch)?;

        // Simulate the batch collation function.
        let _stack: Tensor<B, 4> = Tensor::stack(
            batch
                .rows
                .iter()
                .map(|row| {
                    let value = row.get_column::<Tensor<B, 3>>(&schema, "tensor");
                    let value = value.expect("Failed to get tensor from row");
                    value.clone()
                })
                .collect::<Vec<Tensor<B, 3>>>(),
            0,
        );

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
