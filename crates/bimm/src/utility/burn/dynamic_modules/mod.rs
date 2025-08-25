//! # Module Wrapper

use burn::module::{Module, ModuleMapper, ModuleVisitor};
use burn::prelude::Backend;
use burn::record::{PrecisionSettings, Record};
use serde::Serialize;
use serde::de::DeserializeOwned;
use std::any::Any;

pub mod dyn_mapper;
pub mod dyn_visitor;
/*
trait DynModuleInner<B: Backend>: Send {
    fn clone_box(&self) -> Box<dyn DynModuleInner<B>>;
    fn fork_box(self: Box<Self>, device: &B::Device) -> Box<dyn DynModuleInner<B>>;
    fn to_device_box(self: Box<Self>, device: &B::Device) -> Box<dyn DynModuleInner<B>>;
    fn no_grad_box(self: Box<Self>) -> Box<dyn DynModuleInner<B>>;
    fn map_box<Mapper: ModuleMapper<B>>(self: Box<Self>, mapper: &mut Mapper) -> Box<dyn DynModuleInner<B>>;

    fn num_params(&self) -> usize;
    fn collect_devices(&self, devices: Devices<B>) -> Devices<B>;
    fn visit<Visitor: ModuleVisitor<B>>(self: Box<Self>, visitor: &mut Visitor);
}

// Blanket implementation
impl<B: Backend, M> DynModuleInner<B> for M
where
    M: Module<B> + Clone + Send + 'static
{
    fn clone_box(&self) -> Box<dyn DynModuleInner<B>> {
        Box::new(self.clone())
    }

    fn collect_devices(&self, devices: Devices<B>) -> Devices<B> {
        self.collect_devices(devices)
    }

    fn fork_box(self: Box<Self>, device: &B::Device) -> Box<dyn DynModuleInner<B>> {
        // Unbox, fork (consuming), and rebox
        Box::new((*self).fork(device))
    }

    fn to_device_box(self: Box<Self>, device: &B::Device) -> Box<dyn DynModuleInner<B>> {
        Box::new((*self).to_device(device))
    }

    fn no_grad_box(self: Box<Self>) -> Box<dyn DynModuleInner<B>> {
        Box::new((*self).no_grad())
    }

    fn num_params(&self) -> usize {
        self.num_params()
    }
}

pub trait ForwardTtoT {
    fn forward<B: Backend, const D: usize>(
        &self,
        x: Tensor<B, D>
    ) -> Tensor<B, D>;
}
 */

#[cfg(test)]
mod tests {
    #[test]
    fn test_dyn_module_inner() {}
}
