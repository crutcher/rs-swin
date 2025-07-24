use crate::core::{
    BuildOperator, BuildPlan, DataTypeDescription, OpBinding, OperatorSpec, ParameterSpec,
};
use crate::define_operator_id;
use burn::data::dataset::vision::PixelDepth;
use burn::prelude::{Backend, Int, Tensor};
use burn::tensor::TensorData;
use image::{ColorType, DynamicImage};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::BTreeMap;
use std::marker::PhantomData;
use std::sync::Arc;

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

define_operator_id!(IMG_TO_TENSOR);

/// Config for the ImgToTensor operator.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ImgToTensorConfig {}

impl Default for ImgToTensorConfig {
    fn default() -> Self {
        ImgToTensorConfig::new()
    }
}

impl ImgToTensorConfig {
    /// Creates a new `ImgToTensorConfig`.
    pub fn new() -> Self {
        ImgToTensorConfig {}
    }
}

/// Configures the ImgToTensor operator for a specific backend.
pub fn img_to_tensor_op_binding<B: Backend>(device: &B::Device) -> Arc<dyn OpBinding> {
    let spec: OperatorSpec = OperatorSpec::new()
        .with_operator_id(IMG_TO_TENSOR)
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
    _config: ImgToTensorConfig,
    device: B::Device,
}

impl<B: Backend> ImgToTensor<B> {
    /// Creates a new `ImgToTensor` operator.
    pub fn bind_device(
        config: ImgToTensorConfig,
        device: &B::Device,
    ) -> Result<Box<dyn BuildOperator>, String> {
        let op: ImgToTensor<B> = ImgToTensor {
            _config: config,
            device: device.clone(),
        };
        let b = Box::new(op);
        Ok(b)
    }
}

impl<B: Backend> BuildOperator for ImgToTensor<B> {
    fn apply(
        &self,
        inputs: &BTreeMap<&str, Option<&dyn Any>>,
    ) -> Result<BTreeMap<String, Option<Arc<dyn Any>>>, String> {
        let image = inputs
            .get("image")
            .and_then(|v| v.as_ref())
            .and_then(|v| v.downcast_ref::<DynamicImage>())
            .ok_or_else(|| "Expected input 'image' to be of type DynamicImage".to_string())?;

        let x = image.to_rgba8();
        let height = x.height() as usize;
        let width = x.width() as usize;
        let colors = 4;

        let mut data = Vec::with_capacity(height * width * colors);
        x.pixels().for_each(|p| data.extend(p.0));

        let shape = vec![height, width, colors];
        let td: TensorData = TensorData::new(data, shape);

        let tensor: Tensor<B, 3, Int> = Tensor::from_data(td, &self.device);

        let mut result = BTreeMap::new();
        result.insert("tensor".to_string(), Some(Arc::new(tensor) as Arc<dyn Any>));

        Ok(result)
    }
}

type BindDeviceFunc<C, D> = fn(config: C, device: &D) -> Result<Box<dyn BuildOperator>, String>;

/// A binding for the `BurnDeviceOpBinding` that allows it to be used with a specific backend and operator.
pub struct BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    T: BuildOperator,
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
    T: BuildOperator,
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

impl<B, T, C> OpBinding for BurnDeviceOpBinding<B, T, C>
where
    B: Backend,
    C: DeserializeOwned,
    T: BuildOperator,
{
    fn spec(&self) -> &OperatorSpec {
        &self.spec
    }

    fn validate_build_plan(
        &self,
        build_plan: &BuildPlan,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<(), String> {
        self.init_operator(build_plan, input_types, output_types)
            .map(|_| ())
    }

    fn init_operator(
        &self,
        build_plan: &BuildPlan,
        input_types: &BTreeMap<String, DataTypeDescription>,
        output_types: &BTreeMap<String, DataTypeDescription>,
    ) -> Result<Box<dyn BuildOperator>, String> {
        self.spec.validate(input_types, output_types)?;

        let config = serde_json::from_value(build_plan.config.clone()).map_err(|_| {
            format!(
                "Invalid config: {}",
                serde_json::to_string_pretty(&build_plan.config).unwrap()
            )
        })?;

        let loader = (self.bind_device)(config, &self.device)?;

        Ok(loader)
    }
}
