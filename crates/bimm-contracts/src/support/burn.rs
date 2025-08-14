//! Burn Framework Support

use crate::ShapeArgument;
use alloc::vec::Vec;
use burn::prelude::{Backend, Shape, Tensor};
use burn::tensor::BasicOps;

impl ShapeArgument for &Shape {
    fn get_shape_vec(self) -> Vec<usize> {
        self.dims.clone()
    }
}

impl ShapeArgument for Shape {
    fn get_shape_vec(self) -> Vec<usize> {
        self.dims
    }
}

impl<B, const D: usize, K> ShapeArgument for &Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn get_shape_vec(self) -> Vec<usize> {
        self.shape().dims.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use alloc::vec;

    #[test]
    fn test_shape_argument() {
        let expected = vec![2, 3, 4];

        let shape = Shape::from([2, 3, 4]);
        assert_eq!(&shape.clone().get_shape_vec(), &expected);

        let shape_ref: &Shape = &shape;
        assert_eq!(&shape_ref.get_shape_vec(), &expected);

        let tensor: Tensor<burn::backend::NdArray, 2> = Tensor::zeros([2, 2], &Default::default());
        let tensor_ref = &tensor;
        assert_eq!(&tensor_ref.get_shape_vec(), &[2, 2]);
    }
}
