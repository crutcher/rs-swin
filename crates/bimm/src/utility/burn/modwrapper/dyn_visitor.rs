//! # Burn [`ModuleVisitor`] Dynamic Wrapper Support

use burn::module::{ModuleMapper, ModuleVisitor, ParamId};
use burn::prelude::{Backend, Tensor};
use std::any::Any;
use std::marker::PhantomData;

/// Wraps a [`ModuleVisitor`] as a [`DynModuleVisitor`] for dynamic dispatch.
///
/// Supports tensor dims from 1 to 7.
pub struct DynModuleVisitorBridge<'a, B, V>
where
    B: Backend,
    V: ModuleVisitor<B>,
{
    inner: &'a mut V,
    _phantom: PhantomData<B>,
}
impl<'a, B, V> DynModuleVisitorBridge<'a, B, V>
where
    B: Backend,
    V: ModuleVisitor<B>,
{
    /// Create a new visitor bridge.
    pub fn new(inner: &'a mut V) -> Self {
        Self {
            inner,
            _phantom: PhantomData,
        }
    }
}

/// The dynamic dispatch trait for [`ModuleVisitor`].
pub trait DynModuleVisitor<B: Backend> {
    /// The dynamic equivalent for [`ModuleVisitor::visit_float`].
    fn visit_float_dyn(
        &mut self,
        id: ParamId,
        tensor: &dyn Any,
    );

    /// The dynamic equivalent for [`ModuleVisitor::visit_int`].
    fn visit_int_dyn(
        &mut self,
        id: ParamId,
        tensor: &dyn Any,
    );

    /// The dynamic equivalent for [`ModuleVisitor::visit_bool`].
    fn visit_bool_dyn(
        &mut self,
        id: ParamId,
        tensor: &dyn Any,
    );
}

macro_rules! _impl_visit_dims {
    (
        $self:ident,
        $id:ident,
        $tensor:ident,
        $kind:ident,
        $method:ident,
        [ $($dim:literal),* ]
    ) => {
        $(
            if let Some(t) = $tensor.downcast_ref::<Tensor<B, $dim, $kind>>() {
                $self.inner.$method($id, t);
                return
            }
        )*
    };
}
impl<'a, B: Backend, V: ModuleVisitor<B>> DynModuleVisitor<B> for DynModuleVisitorBridge<'a, B, V> {
    fn visit_float_dyn(
        &mut self,
        id: ParamId,
        tensor: &dyn Any,
    ) {
        use burn::prelude::{Float, Tensor};
        _impl_visit_dims!(self, id, tensor, Float, visit_float, [1, 2, 3, 4, 5, 6, 7]);
        panic!("Unsupported tensor type/dims");
    }

    fn visit_int_dyn(
        &mut self,
        id: ParamId,
        tensor: &dyn Any,
    ) {
        use burn::prelude::{Int, Tensor};
        _impl_visit_dims!(self, id, tensor, Int, visit_int, [1, 2, 3, 4, 5, 6, 7]);
        panic!("Unsupported tensor type/dims");
    }

    fn visit_bool_dyn(
        &mut self,
        id: ParamId,
        tensor: &dyn Any,
    ) {
        use burn::prelude::{Bool, Tensor};
        _impl_visit_dims!(self, id, tensor, Bool, visit_bool, [1, 2, 3, 4, 5, 6, 7]);
        panic!("Unsupported tensor type/dims");
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    pub struct TestVisitor {
        floats: Vec<VisitRecord>,
        ints: Vec<VisitRecord>,
        bools: Vec<VisitRecord>,
    }
    impl TestVisitor {
        pub fn new() -> Self {
            Self {
                floats: Vec::new(),
                ints: Vec::new(),
                bools: Vec::new(),
            }
        }
    }
    impl<B: Backend> ModuleVisitor<B> for TestVisitor {
        fn visit_float<const D: usize>(
            &mut self,
            id: ParamId,
            tensor: &Tensor<B, D>,
        ) {
            self.floats.push(VisitRecord::new(id, tensor.shape()))
        }
        fn visit_int<const D: usize>(
            &mut self,
            id: ParamId,
            tensor: &Tensor<B, D, Int>,
        ) {
            self.ints.push(VisitRecord::new(id, tensor.shape()))
        }
        fn visit_bool<const D: usize>(
            &mut self,
            id: ParamId,
            tensor: &Tensor<B, D, Bool>,
        ) {
            self.bools.push(VisitRecord::new(id, tensor.shape()))
        }
    }

    #[test]
    fn test_visitor_bridge() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut visitor = TestVisitor::new();
        let mut bridge = DynModuleVisitorBridge::<B, TestVisitor>::new(&mut visitor);

        bridge.visit_float_dyn(
            ParamId::from(101),
            &Tensor::<B, 1, Float>::empty([1], &device),
        );
        bridge.visit_float_dyn(
            ParamId::from(102),
            &Tensor::<B, 2, Float>::empty([1, 1], &device),
        );
        bridge.visit_float_dyn(
            ParamId::from(103),
            &Tensor::<B, 3, Float>::empty([1, 1, 1], &device),
        );
        bridge.visit_float_dyn(
            ParamId::from(104),
            &Tensor::<B, 4, Float>::empty([1, 1, 1, 1], &device),
        );
        bridge.visit_float_dyn(
            ParamId::from(105),
            &Tensor::<B, 5, Float>::empty([1, 1, 1, 1, 1], &device),
        );
        bridge.visit_float_dyn(
            ParamId::from(106),
            &Tensor::<B, 6, Float>::empty([1, 1, 1, 1, 1, 1], &device),
        );
        bridge.visit_float_dyn(
            ParamId::from(107),
            &Tensor::<B, 7, Float>::empty([1, 1, 1, 1, 1, 1, 1], &device),
        );

        bridge.visit_int_dyn(
            ParamId::from(201),
            &Tensor::<B, 1, Int>::empty([1], &device),
        );
        bridge.visit_int_dyn(
            ParamId::from(202),
            &Tensor::<B, 2, Int>::empty([1, 1], &device),
        );
        bridge.visit_int_dyn(
            ParamId::from(203),
            &Tensor::<B, 3, Int>::empty([1, 1, 1], &device),
        );
        bridge.visit_int_dyn(
            ParamId::from(204),
            &Tensor::<B, 4, Int>::empty([1, 1, 1, 1], &device),
        );
        bridge.visit_int_dyn(
            ParamId::from(205),
            &Tensor::<B, 5, Int>::empty([1, 1, 1, 1, 1], &device),
        );
        bridge.visit_int_dyn(
            ParamId::from(206),
            &Tensor::<B, 6, Int>::empty([1, 1, 1, 1, 1, 1], &device),
        );
        bridge.visit_int_dyn(
            ParamId::from(207),
            &Tensor::<B, 7, Int>::empty([1, 1, 1, 1, 1, 1, 1], &device),
        );

        bridge.visit_bool_dyn(
            ParamId::from(301),
            &Tensor::<B, 1, Bool>::empty([1], &device),
        );
        bridge.visit_bool_dyn(
            ParamId::from(302),
            &Tensor::<B, 2, Bool>::empty([1, 1], &device),
        );
        bridge.visit_bool_dyn(
            ParamId::from(303),
            &Tensor::<B, 3, Bool>::empty([1, 1, 1], &device),
        );
        bridge.visit_bool_dyn(
            ParamId::from(304),
            &Tensor::<B, 4, Bool>::empty([1, 1, 1, 1], &device),
        );
        bridge.visit_bool_dyn(
            ParamId::from(305),
            &Tensor::<B, 5, Bool>::empty([1, 1, 1, 1, 1], &device),
        );
        bridge.visit_bool_dyn(
            ParamId::from(306),
            &Tensor::<B, 6, Bool>::empty([1, 1, 1, 1, 1, 1], &device),
        );
        bridge.visit_bool_dyn(
            ParamId::from(307),
            &Tensor::<B, 7, Bool>::empty([1, 1, 1, 1, 1, 1, 1], &device),
        );

        assert_eq!(
            &visitor.floats,
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
            &visitor.ints,
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
            &visitor.bools,
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
    fn test_visitor_float_too_many_dims() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut visitor = TestVisitor::new();
        let mut bridge = DynModuleVisitorBridge::<B, TestVisitor>::new(&mut visitor);

        let t: Tensor<B, 8> = Tensor::empty([1, 1, 1, 1, 1, 1, 1, 1], &device);
        bridge.visit_float_dyn(ParamId::from(1), &t);
    }

    #[test]
    #[should_panic(expected = "Unsupported tensor type/dims")]
    fn test_visitor_int_too_many_dims() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut visitor = TestVisitor::new();
        let mut bridge = DynModuleVisitorBridge::<B, TestVisitor>::new(&mut visitor);

        let t: Tensor<B, 8, Int> = Tensor::empty([1, 1, 1, 1, 1, 1, 1, 1], &device);
        bridge.visit_int_dyn(ParamId::from(1), &t);
    }

    #[test]
    #[should_panic(expected = "Unsupported tensor type/dims")]
    fn test_visitor_bool_too_many_dims() {
        type B = NdArray;
        let device = NdArrayDevice::default();

        let mut visitor = TestVisitor::new();
        let mut bridge = DynModuleVisitorBridge::<B, TestVisitor>::new(&mut visitor);

        let t: Tensor<B, 8, Bool> = Tensor::empty([1, 1, 1, 1, 1, 1, 1, 1], &device);
        bridge.visit_bool_dyn(ParamId::from(1), &t);
    }
}
