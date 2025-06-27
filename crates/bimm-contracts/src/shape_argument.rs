use burn::prelude::{Backend, Shape, Tensor};
use burn::tensor::BasicOps;

pub trait ShapeArgument {
    fn get_shape(self) -> Shape;
}

impl ShapeArgument for &Shape {
    fn get_shape(self) -> Shape {
        self.clone()
    }
}

impl ShapeArgument for Shape {
    fn get_shape(self) -> Shape {
        self
    }
}

impl<const D: usize> ShapeArgument for &[usize; D] {
    fn get_shape(self) -> Shape {
        Shape::from(*self)
    }
}

impl<const D: usize> ShapeArgument for &[u32; D] {
    fn get_shape(self) -> Shape {
        Shape::from(self.map(|d| d as usize))
    }
}

impl<const D: usize> ShapeArgument for &[i32; D] {
    fn get_shape(self) -> Shape {
        Shape::from(self.map(|d| d as usize))
    }
}

impl ShapeArgument for &[usize] {
    fn get_shape(self) -> Shape {
        Shape::from(self.to_vec())
    }
}

impl ShapeArgument for &[u32] {
    fn get_shape(self) -> Shape {
        Shape::from(self.iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

impl ShapeArgument for &[i32] {
    fn get_shape(self) -> Shape {
        Shape::from(self.iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

impl ShapeArgument for &Vec<usize> {
    fn get_shape(self) -> Shape {
        Shape::from(self)
    }
}

impl ShapeArgument for &Vec<u32> {
    fn get_shape(self) -> Shape {
        Shape::from(self.iter().map(|&d| d as usize).collect::<Vec<_>>())
    }
}

impl<B, const D: usize, K> ShapeArgument for &Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn get_shape(self) -> Shape {
        self.shape()
    }
}
