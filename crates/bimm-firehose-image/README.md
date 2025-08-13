# bimm-firehose-image

Image processing support for `bimm-firehose`.

### Example

This is an example of how to use the image processing stages.

```rust
use bimm_firehose_image::{
    AugmentImageOperation, ChooseOneStage, HorizontalFlipStage, SpeckleStage,
    WithProbStage,
};
use bimm_firehose_schema::{AugmentColumn, AugmentSchema, FirehoseEnv};
use bimm_firehose_utils::FirehoseError;
use std::sync::Arc;

const PATH_COLUMN: &str = "path";
const CLASS_COLUMN: &str = "class";
const SEED_COLUMN: &str = "seed";
const IMAGE_COLUMN: &str = "image";
const AUG_COLUMN: &str = "aug";

fn example() {
    let firehose_env = Arc::new(init_default_operator_environment());

    let mut schema = FirehoseTableSchema::from_columns(&[
        ColumnSchema::new::<String>(PATH_COLUMN).with_description("path to the image"),
        ColumnSchema::new::<i32>(CLASS_COLUMN).with_description("image class"),
        ColumnSchema::new::<u64>(SEED_COLUMN).with_description("instance rng seed"),
    ]);
    
    ImageLoader::default()
        .with_resize(ResizeSpec::new(ImageShape {
            width: 32,
            height: 32,
        }))
        .with_recolor(ColorType::Rgb8)
        .to_plan(PATH_COLUMN, IMAGE_COLUMN)
        .apply_to_schema(&mut schema, firehose_env.as_ref())?;
    
    AugmentImageOperation::new(vec![
        Arc::new(WithProbStage::new(
            0.5,
            Arc::new(HorizontalFlipStage::new()),
        )),
        Arc::new(WithProbStage::new(0.5, Arc::new(SpeckleStage::default()))),
        Arc::new(
            ChooseOneStage::new()
                .with_choice(1.0, Arc::new(BlurStage::Gaussian { sigma: 0.5 }))
                .with_choice(1.0, Arc::new(BlurStage::Gaussian { sigma: 1.0 }))
                .with_choice(1.0, Arc::new(BlurStage::Gaussian { sigma: 1.5 }))
                .with_noop_weight(6.0),
        ),
    ])
    .to_plan(SEED_COLUMN, IMAGE_COLUMN, AUG_COLUMN)
    .apply_to_schema(&mut schema, firehose_env.as_ref())?;
}
```