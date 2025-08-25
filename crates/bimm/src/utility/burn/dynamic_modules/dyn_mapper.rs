//! # Burn [`ModuleMapper`] Dynamic Wrapper Support
use burn::module::{ModuleMapper, ModuleVisitor, ParamId};
use burn::prelude::Backend;
use std::any::Any;
use std::marker::PhantomData;

/// Wraps a [`ModuleMapper`] as a [`DynModuleMapper`] for dynamic dispatch.
///
/// Supports tensor dims from 1 to 7.
pub struct DynModuleMapperBridge<'a, B, M>
where
    B: Backend,
    M: ModuleMapper<B>,
{
    inner: &'a mut M,
    _phantom: PhantomData<B>,
}

impl<'a, B, M> DynModuleMapperBridge<'a, B, M>
where
    B: Backend,
    M: ModuleMapper<B>,
{
    /// Create a new visitor bridge.
    pub fn new(inner: &'a mut M) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

/// Backend for [`DynModuleMapperBridge`]
pub trait DynModuleMapper<B: Backend> {
    /// The dynamic equivalent for [`ModuleMapper::map_float`].
    fn map_float_dyn(
        &mut self,
        id: ParamId,
        tensor: Box<dyn Any>,
    ) -> Box<dyn Any>;

    /// The dynamic equivalent for [`ModuleMapper::map_int`].
    fn map_int_dyn(
        &mut self,
        id: ParamId,
        tensor: Box<dyn Any>,
    ) -> Box<dyn Any>;

    /// The dynamic equivalent for [`ModuleMapper::map_bool`].
    fn map_bool_dyn(
        &mut self,
        id: ParamId,
        tensor: Box<dyn Any>,
    ) -> Box<dyn Any>;
}

macro_rules! _impl_map_dims {
    (
        $self:ident,
        $id:ident,
        $tensor:ident,
        $kind:ident,
        $method:ident,
        [ $($dim:literal),* ]) => {
        $(
            if let Some(t) = $tensor.downcast_ref::<Tensor<B, $dim, $kind>>() {
              return Box::new($ self.inner.$method($id, t.clone()));
            }
        )*
    };
}

impl<'a, B: Backend, M: ModuleMapper<B>> DynModuleMapper<B> for DynModuleMapperBridge<'a, B, M> {
    fn map_float_dyn(
        &mut self,
        id: ParamId,
        tensor: Box<dyn Any>,
    ) -> Box<dyn Any> {
        use burn::prelude::{Float, Tensor};
        _impl_map_dims!(self, id, tensor, Float, map_float, [1, 2, 3, 4, 5, 6, 7]);
        panic!("Unsupported tensor type/dims");
    }

    fn map_int_dyn(
        &mut self,
        id: ParamId,
        tensor: Box<dyn Any>,
    ) -> Box<dyn Any> {
        use burn::prelude::{Int, Tensor};
        _impl_map_dims!(self, id, tensor, Int, map_int, [1, 2, 3, 4, 5, 6, 7]);
        panic!("Unsupported tensor type/dims");
    }

    fn map_bool_dyn(
        &mut self,
        id: ParamId,
        tensor: Box<dyn Any>,
    ) -> Box<dyn Any> {
        use burn::prelude::{Bool, Tensor};
        _impl_map_dims!(self, id, tensor, Bool, map_bool, [1, 2, 3, 4, 5, 6, 7]);
        panic!("Unsupported tensor type/dims");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utility;
    use crate::utility::burn::dynamic_modules::dyn_visitor::{
        DynModuleVisitor, DynModuleVisitorBridge,
    };
    use burn::backend::NdArray;
    use burn::backend::ndarray::NdArrayDevice;
    use burn::prelude::{Bool, Float, Int, Shape, Tensor};

    #[derive(Debug, PartialEq)]
    pub struct VisitRecord {
        pub id: ParamId,
        pub shape: Shape,
    }
    impl VisitRecord {
        pub fn new(
            id: ParamId,
            shape: Shape,
        ) -> Self {
            Self { id, shape }
        }
    }

    pub struct TestMapper {
        floats: Vec<VisitRecord>,
        ints: Vec<VisitRecord>,
        bools: Vec<VisitRecord>,
    }
    impl TestMapper {
        fn new() -> Self {
            Self {
                floats: Vec::new(),
                ints: Vec::new(),
                bools: Vec::new(),
            }
        }
    }

    impl<B: Backend> ModuleMapper<B> for TestMapper {
        fn map_float<const D: usize>(
            &mut self,
            id: ParamId,
            tensor: Tensor<B, D>,
        ) -> Tensor<B, D> {
            self.floats.push(VisitRecord::new(id, tensor.shape()));
            tensor
        }

        fn map_int<const D: usize>(
            &mut self,
            id: ParamId,
            tensor: Tensor<B, D, Int>,
        ) -> Tensor<B, D, Int> {
            self.ints.push(VisitRecord::new(id, tensor.shape()));
            tensor
        }

        fn map_bool<const D: usize>(
            &mut self,
            id: ParamId,
            tensor: Tensor<B, D, Bool>,
        ) -> Tensor<B, D, Bool> {
            self.bools.push(VisitRecord::new(id, tensor.shape()));
            tensor
        }
    }

    #[test]
    fn test_mapper_bridge() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut mapper = TestMapper::new();
        let mut bridge: DynModuleMapperBridge<B, TestMapper> =
            DynModuleMapperBridge::new(&mut mapper);

        let tensor: Tensor<B, 1, Float> = Tensor::empty([1], &device);
        let result = bridge.map_float_dyn(ParamId::from(101), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 1, Float>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 2, Float> = Tensor::empty([1, 1], &device);
        let result = bridge.map_float_dyn(ParamId::from(102), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 2, Float>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 3, Float> = Tensor::empty([1, 1, 1], &device);
        let result = bridge.map_float_dyn(ParamId::from(103), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 3, Float>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 4, Float> = Tensor::empty([1, 1, 1, 1], &device);
        let result = bridge.map_float_dyn(ParamId::from(104), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 4, Float>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 5, Float> = Tensor::empty([1, 1, 1, 1, 1], &device);
        let result = bridge.map_float_dyn(ParamId::from(105), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 5, Float>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 6, Float> = Tensor::empty([1, 1, 1, 1, 1, 1], &device);
        let result = bridge.map_float_dyn(ParamId::from(106), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 6, Float>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 7, Float> = Tensor::empty([1, 1, 1, 1, 1, 1, 1], &device);
        let result = bridge.map_float_dyn(ParamId::from(107), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 7, Float>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 1, Int> = Tensor::empty([1], &device);
        let result = bridge.map_int_dyn(ParamId::from(201), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 1, Int>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 2, Int> = Tensor::empty([1, 1], &device);
        let result = bridge.map_int_dyn(ParamId::from(202), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 2, Int>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 3, Int> = Tensor::empty([1, 1, 1], &device);
        let result = bridge.map_int_dyn(ParamId::from(203), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 3, Int>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 4, Int> = Tensor::empty([1, 1, 1, 1], &device);
        let result = bridge.map_int_dyn(ParamId::from(204), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 4, Int>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 5, Int> = Tensor::empty([1, 1, 1, 1, 1], &device);
        let result = bridge.map_int_dyn(ParamId::from(205), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 5, Int>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 6, Int> = Tensor::empty([1, 1, 1, 1, 1, 1], &device);
        let result = bridge.map_int_dyn(ParamId::from(206), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 6, Int>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 7, Int> = Tensor::empty([1, 1, 1, 1, 1, 1, 1], &device);
        let result = bridge.map_int_dyn(ParamId::from(207), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 7, Int>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 1, Bool> = Tensor::empty([1], &device);
        let result = bridge.map_bool_dyn(ParamId::from(301), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 1, Bool>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 2, Bool> = Tensor::empty([1, 1], &device);
        let result = bridge.map_bool_dyn(ParamId::from(302), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 2, Bool>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 3, Bool> = Tensor::empty([1, 1, 1], &device);
        let result = bridge.map_bool_dyn(ParamId::from(303), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 3, Bool>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 4, Bool> = Tensor::empty([1, 1, 1, 1], &device);
        let result = bridge.map_bool_dyn(ParamId::from(304), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 4, Bool>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 5, Bool> = Tensor::empty([1, 1, 1, 1, 1], &device);
        let result = bridge.map_bool_dyn(ParamId::from(305), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 5, Bool>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 6, Bool> = Tensor::empty([1, 1, 1, 1, 1, 1], &device);
        let result = bridge.map_bool_dyn(ParamId::from(306), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 6, Bool>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        let tensor: Tensor<B, 7, Bool> = Tensor::empty([1, 1, 1, 1, 1, 1, 1], &device);
        let result = bridge.map_bool_dyn(ParamId::from(307), Box::new(tensor.clone()));
        result
            .downcast_ref::<Tensor<B, 7, Bool>>()
            .unwrap()
            .to_data()
            .assert_eq(&tensor.to_data(), true);

        assert_eq!(
            &mapper.floats,
            &vec![
                VisitRecord::new(ParamId::from(101), Shape::from([1])),
                VisitRecord::new(ParamId::from(102), Shape::from([1, 1])),
                VisitRecord::new(ParamId::from(103), Shape::from([1, 1, 1])),
                VisitRecord::new(ParamId::from(104), Shape::from([1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(105), Shape::from([1, 1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(106), Shape::from([1, 1, 1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(107), Shape::from([1, 1, 1, 1, 1, 1, 1])),
            ],
        );
        assert_eq!(
            &mapper.ints,
            &vec![
                VisitRecord::new(ParamId::from(201), Shape::from([1])),
                VisitRecord::new(ParamId::from(202), Shape::from([1, 1])),
                VisitRecord::new(ParamId::from(203), Shape::from([1, 1, 1])),
                VisitRecord::new(ParamId::from(204), Shape::from([1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(205), Shape::from([1, 1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(206), Shape::from([1, 1, 1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(207), Shape::from([1, 1, 1, 1, 1, 1, 1])),
            ],
        );
        assert_eq!(
            &mapper.bools,
            &vec![
                VisitRecord::new(ParamId::from(301), Shape::from([1])),
                VisitRecord::new(ParamId::from(302), Shape::from([1, 1])),
                VisitRecord::new(ParamId::from(303), Shape::from([1, 1, 1])),
                VisitRecord::new(ParamId::from(304), Shape::from([1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(305), Shape::from([1, 1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(306), Shape::from([1, 1, 1, 1, 1, 1])),
                VisitRecord::new(ParamId::from(307), Shape::from([1, 1, 1, 1, 1, 1, 1])),
            ],
        );
    }

    #[test]
    #[should_panic(expected = "Unsupported tensor type/dims")]
    fn test_mapper_float_too_many_dims() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut mapper = TestMapper::new();
        let mut bridge: DynModuleMapperBridge<B, TestMapper> =
            DynModuleMapperBridge::new(&mut mapper);

        let t: Tensor<B, 8> = Tensor::empty([1, 1, 1, 1, 1, 1, 1, 1], &device);
        let _t = bridge.map_float_dyn(ParamId::from(1), Box::new(t));
    }

    #[test]
    #[should_panic(expected = "Unsupported tensor type/dims")]
    fn test_mapper_int_too_many_dims() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut mapper = TestMapper::new();
        let mut bridge: DynModuleMapperBridge<B, TestMapper> =
            DynModuleMapperBridge::new(&mut mapper);

        let t: Tensor<B, 8, Int> = Tensor::empty([1, 1, 1, 1, 1, 1, 1, 1], &device);
        let _t = bridge.map_int_dyn(ParamId::from(1), Box::new(t));
    }

    #[test]
    #[should_panic(expected = "Unsupported tensor type/dims")]
    fn test_mapper_bool_too_many_dims() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut mapper = TestMapper::new();
        let mut bridge: DynModuleMapperBridge<B, TestMapper> =
            DynModuleMapperBridge::new(&mut mapper);

        let t: Tensor<B, 8, Bool> = Tensor::empty([1, 1, 1, 1, 1, 1, 1, 1], &device);
        let _t = bridge.map_bool_dyn(ParamId::from(1), Box::new(t));
    }
}
