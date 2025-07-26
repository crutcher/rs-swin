use crate::core::{ColumnBuildOperator, ColumnBuildOperatorBuilder, OperatorSpec, ParameterSpec, ColumnBuildOperationInitContext};
use crate::define_operator_id;
use burn::data::dataset::vision::PixelDepth;
use burn::prelude::{Backend, Tensor};
use burn::tensor::{TensorData, f16};
use image::{ColorType, DynamicImage};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

define_operator_id!(IMAGE_TO_TENSOR);

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

/// Config for the ImgToTensor operator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImgToTensorConfig {
    /// The target data type for the tensor.
    #[serde(default)]
    pub dtype: TargetDType,
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
}

/// Configures the ImgToTensor operator for a specific backend.
pub fn img_to_tensor_op_binding<B: Backend>(device: &B::Device) -> Arc<dyn ColumnBuildOperatorBuilder> {
    let spec: OperatorSpec = OperatorSpec::new()
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
pub struct ImgToTensor<B: Backend> {
    config: ImgToTensorConfig,
    device: B::Device,
}

impl<B: Backend> ImgToTensor<B> {
    /// Creates a new `ImgToTensor` operator.
    pub fn bind_device(
        config: ImgToTensorConfig,
        device: &B::Device,
    ) -> Result<Box<dyn ColumnBuildOperator>, String> {
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
fn image_to_f32_tensor<B: Backend>(
    image: &DynamicImage,
    device: &B::Device,
) -> Tensor<B, 3> {
    let height = image.height() as usize;
    let width = image.width() as usize;
    let colors = image.color().channel_count() as usize;
    let shape = vec![height, width, colors];

    let pixvec = image_to_pixvec(image);
    let data: Vec<f32> = pixvec
        .iter()
        .map(|p| pixel_depth_to_f32(p.clone()))
        .collect();

    Tensor::from_data_dtype(
        TensorData::new(data, shape),
        device,
        burn::tensor::DType::F32,
    )
}

impl<B: Backend> ColumnBuildOperator for ImgToTensor<B> {
    fn apply(
        &self,
        inputs: &BTreeMap<&str, Option<&dyn Any>>,
    ) -> Result<BTreeMap<String, Option<Arc<dyn Any>>>, String> {
        let image = inputs
            .get("image")
            .and_then(|v| v.as_ref())
            .and_then(|v| v.downcast_ref::<DynamicImage>())
            .ok_or_else(|| "Expected input 'image' to be of type DynamicImage".to_string())?;

        let mut result = BTreeMap::new();
        match self.config.dtype {
            TargetDType::F32 => {
                let tensor: Tensor<B, 3> = image_to_f32_tensor(image, &self.device);
                result.insert("tensor".to_string(), Some(Arc::new(tensor) as Arc<dyn Any>));
            }
        }

        Ok(result)
    }
}

type BindDeviceFunc<C, D> = fn(config: C, device: &D) -> Result<Box<dyn ColumnBuildOperator>, String>;

/// A binding for the `BurnDeviceOpBinding` that allows it to be used with a specific backend and operator.
pub struct BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    T: ColumnBuildOperator,
    C: DeserializeOwned,
{
    spec: OperatorSpec,
    device: B::Device,
    bind_device: BindDeviceFunc<C, B::Device>,
    phantom: PhantomData<T>,
}

impl<B, T, C> BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    C: DeserializeOwned,
    T: ColumnBuildOperator,
{
    /// Creates a new `BurnDeviceOpBinding`.
    pub fn new(
        spec: OperatorSpec,
        device: &B::Device,
        bind_device: BindDeviceFunc<C, B::Device>,
    ) -> Self {
        BurnDeviceOpBinding {
            device: device.clone(),
            spec,
            bind_device,
            phantom: PhantomData,
        }
    }
}

impl<B, T, C> ColumnBuildOperatorBuilder for BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    C: DeserializeOwned,
    T: ColumnBuildOperator,
{
    fn spec(&self) -> &OperatorSpec {
        &self.spec
    }

    fn validate(&self, context: &ColumnBuildOperationInitContext) -> Result<(), String> {
        self.build(context).map(|_| ())
    }

    fn build(&self, context: &ColumnBuildOperationInitContext) -> Result<Box<dyn ColumnBuildOperator>, String> {
        self.spec.validate(context.input_types(), context.output_types())?;

        let config = &context.build_plan().config;
        let config = serde_json::from_value(config.clone()).map_err(
            |_| {
                format!(
                    "Invalid config: {}",
                    serde_json::to_string_pretty(config).unwrap()
                )
            },
        )?;

        let loader = (self.bind_device)(config, &self.device)?;
        Ok(loader)
    }
}
