use crate::core::operations::factory::{FirehoseOperatorFactory, FirehoseOperatorInitContext};
use crate::core::operations::operator::FirehoseOperator;
use crate::core::operations::planner::OperationPlan;
use crate::core::operations::signature::{FirehoseOperatorSignature, ParameterSpec};
use crate::core::rows::FirehoseRowTransaction;
use crate::core::{FirehoseRowReader, FirehoseRowWriter, ValueBox};
use crate::define_firehose_operator_id;
use anyhow::Context;
use burn::data::dataset::vision::PixelDepth;
use burn::prelude::{Backend, Tensor, Int};
use burn::tensor::{TensorData, f16, DType};
use image::{ColorType, DynamicImage};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::sync::Arc;

define_firehose_operator_id!(IMAGE_TO_TENSOR);

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

/// Target dtype for the tensor conversion.
#[derive(Debug, Default, Clone, Copy, Deserialize, Serialize)]
pub enum TargetDType {
    /// Convert to `[0.0, 1.0]` Float `F32` tensor.
    #[default]
    F32,
}

/// Converts a `PixelDepth` to a `u8`.
pub fn pixel_depth_to_u8(p: PixelDepth) -> u8 {
    match p {
        PixelDepth::U8(v) => v,
        PixelDepth::U16(v) => (v >> 8) as u8, // Convert U16 to U8 by taking the high byte
        PixelDepth::F32(v) => (v * 255.0) as u8, // Scale F32 to U8
    }
}

/// Converts a `PixelDepth` to a `u16`.
pub fn pixel_depth_to_u16(p: PixelDepth) -> u16 {
    match p {
        PixelDepth::U8(v) => (v as u16) << 8, // Convert U8 to U16 by shifting left
        PixelDepth::U16(v) => v,
        PixelDepth::F32(v) => (v * 65535.0) as u16, // Scale F32 to U16
    }
}

/// Converts a `PixelDepth` to a `f16`.
pub fn pixel_depth_to_f16(p: PixelDepth) -> f16 {
    match p {
        PixelDepth::U8(v) => f16::from_f32(v as f32 / 255.0), // Scale U8 to F16
        PixelDepth::U16(v) => f16::from_f32(v as f32 / 65535.0), // Scale U16 to F16
        PixelDepth::F32(v) => f16::from_f32(v),               // Direct conversion for F32
    }
}

/// Converts a `PixelDepth` to a `f32`.
pub fn pixel_depth_to_f32(p: PixelDepth) -> f32 {
    match p {
        PixelDepth::U8(v) => v as f32 / 255.0,    // Scale U8 to F32
        PixelDepth::U16(v) => v as f32 / 65535.0, // Scale U16 to F32
        PixelDepth::F32(v) => v,
    }
}

/// Config for the `ImgToTensor` operator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImgToTensorConfig {
    /// The target data type for the tensor.
    #[serde(default)]
    pub dtype: TargetDType,
    // TODO: add dim-order enum: (HWC vs CHW)
}

impl Default for ImgToTensorConfig {
    fn default() -> Self {
        ImgToTensorConfig::new()
    }
}

impl ImgToTensorConfig {
    /// Creates a new `ImgToTensorConfig`.
    pub fn new() -> Self {
        ImgToTensorConfig {
            dtype: TargetDType::default(),
        }
    }

    /// Extends the configuration to specify the tensor data type.
    pub fn with_dtype(
        self,
        dtype: TargetDType,
    ) -> Self {
        ImgToTensorConfig { dtype }
    }

    /// Converts this configuration to an `OperationPlanner` for the `ImgToTensor` operator.
    ///
    /// # Arguments
    ///
    /// * `image_column` - The name of the input column containing the image.
    /// * `tensor_column` - The name of the output column for the tensor.
    pub fn to_plan(
        self,
        image_column: &str,
        tensor_column: &str,
    ) -> OperationPlan {
        OperationPlan::for_operation_id(IMAGE_TO_TENSOR)
            .with_input("image", image_column)
            .with_output("tensor", tensor_column)
            .with_config(self)
    }
}

/// Configures the `ImgToTensor` operator for a specific backend.
pub fn img_to_tensor_op_binding<B: Backend>(
    device: &B::Device
) -> Arc<dyn FirehoseOperatorFactory> {
    let spec: FirehoseOperatorSignature = FirehoseOperatorSignature::new()
        .with_operator_id(IMAGE_TO_TENSOR)
        .with_description("Converts an image to a tensor.")
        .with_input(
            ParameterSpec::new::<DynamicImage>("image")
                .with_description("Image to convert to a tensor."),
        )
        .with_output(
            ParameterSpec::new::<Tensor<B, 3>>("tensor")
                .with_description("Tensor representation of the image."),
        );

    let func: BindDeviceFunc<ImgToTensorConfig, B::Device> = ImgToTensor::<B>::bind_device;

    Arc::new(BurnDeviceOpBinding::<B, ImgToTensor<B>, ImgToTensorConfig>::new(spec, device, func))
}

/// Operator that converts an image to a tensor.
#[derive(Debug)]
pub struct ImgToTensor<B: Backend> {
    /// The configuration for the operator.
    config: ImgToTensorConfig,

    /// The device on which the tensor will be created.
    device: B::Device,
}

impl<B: Backend> ImgToTensor<B> {
    /// Creates a new `ImgToTensor` operator.
    pub fn bind_device(
        config: ImgToTensorConfig,
        device: &B::Device,
    ) -> anyhow::Result<Box<dyn FirehoseOperator>> {
        let op: ImgToTensor<B> = ImgToTensor {
            config,
            device: device.clone(),
        };
        let b = Box::new(op);
        Ok(b)
    }
}

/// Converts an image to a tensor `[h, w, c]` Float tensor of type `f32`.
///
/// # Arguments
///
/// * `image` - The image to convert.
/// * `device` - The device to create the tensor on.
///
/// # Returns
///
/// A tensor representation of the image with shape `[height, width, channels]`.
pub fn image_to_f32_tensor<B: Backend>(
    image: &DynamicImage,
    device: &B::Device,
) -> Tensor<B, 3> {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let colors = image.color().channel_count() as usize;
    let shape = vec![height, width, colors];

    let pixvec = image_to_pixvec(image);
    let data: Vec<u8> = pixvec
        .iter()
        .map(|p| pixel_depth_to_u8(p.clone()))
        .collect();

    let tensor: Tensor<B, 3, Int> = Tensor::from_data_dtype(
        TensorData::from_bytes(data, shape, DType::U8),
        device,
        DType::U8,
    );

    tensor.float() / 255.0
}

impl<B: Backend> FirehoseOperator for ImgToTensor<B> {
    fn apply_to_row(
        &self,
        txn: &mut FirehoseRowTransaction,
    ) -> anyhow::Result<()> {
        let image = txn.get("image").unwrap().as_ref::<DynamicImage>()?;

        match self.config.dtype {
            TargetDType::F32 => {
                let tensor: Tensor<B, 3> = image_to_f32_tensor(image, &self.device);

                txn.set("tensor", ValueBox::boxing(tensor));
            }
        }

        Ok(())
    }
}

/// Function type for binding a device with an operator configuration.
type BindDeviceFunc<C, D> = fn(config: C, device: &D) -> anyhow::Result<Box<dyn FirehoseOperator>>;

/// A binding for the `BurnDeviceOpBinding` that allows it to be used with a specific backend and operator.
pub struct BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    T: FirehoseOperator,
    C: DeserializeOwned,
{
    /// The operator signature.
    signature: FirehoseOperatorSignature,

    /// The device on which the operator will run.
    device: B::Device,

    /// The function to bind the device with the operator configuration.
    bind_device: BindDeviceFunc<C, B::Device>,

    /// Phantom data to ensure the type parameters are used.
    phantom: PhantomData<T>,
}

impl<B, T, C> Debug for BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    T: FirehoseOperator,
    C: DeserializeOwned,
{
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        f.debug_struct("BurnDeviceOpBinding")
            .field("signature", &self.signature)
            .field("device", &self.device)
            .finish()
    }
}

impl<B, T, C> BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    C: DeserializeOwned,
    T: FirehoseOperator,
{
    /// Creates a new `BurnDeviceOpBinding`.
    ///
    /// # Arguments
    ///
    /// * `signature` - The operator signature.
    /// * `device` - The device on which the operator will run.
    /// * `bind_device` - The function to bind the device with the operator configuration.
    pub fn new(
        signature: FirehoseOperatorSignature,
        device: &B::Device,
        bind_device: BindDeviceFunc<C, B::Device>,
    ) -> Self {
        BurnDeviceOpBinding {
            device: device.clone(),
            signature,
            bind_device,
            phantom: PhantomData,
        }
    }
}

impl<B, T, C> FirehoseOperatorFactory for BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    C: DeserializeOwned,
    T: FirehoseOperator,
{
    fn signature(&self) -> &FirehoseOperatorSignature {
        &self.signature
    }

    fn init(
        &self,
        context: &dyn FirehoseOperatorInitContext,
    ) -> anyhow::Result<Box<dyn FirehoseOperator>> {
        let config = &context.build_plan().config;
        let config = serde_json::from_value(config.clone()).with_context(|| {
            format!(
                "Failed to deserialize operator config for {}: {}",
                self.signature.operator_id.as_deref().unwrap_or("unknown"),
                serde_json::to_string_pretty(config).unwrap()
            )
        })?;

        let loader = (self.bind_device)(config, &self.device)?;
        Ok(loader)
    }
}
