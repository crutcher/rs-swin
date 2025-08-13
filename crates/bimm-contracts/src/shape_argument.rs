use burn::prelude::{Backend, Shape, Tensor};
use burn::tensor::BasicOps;

/// A trait that provides a method to extract the shape from various types.
pub trait ShapeArgument {
    /// Extracts the shape from the implementing type.
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

impl ShapeArgument for &Vec<i32> {
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

#[cfg(test)]
mod tests {
    use super::*;
    use burn::prelude::Shape;

    #[test]
    fn test_shape_argument() {
        let shape = Shape::from([2, 3, 4]);
        assert_eq!(shape.clone().get_shape(), shape);

        let shape_ref: &Shape = &shape;
        assert_eq!(shape_ref.get_shape(), shape);

        {
            let arr: [usize; 3] = [2, 3, 4];
            assert_eq!(arr.get_shape(), shape);

            let arr_ref: &[usize] = &arr;
            assert_eq!(arr_ref.get_shape(), shape);
        }

        {
            let arr: [u32; 3] = [2, 3, 4];
            assert_eq!(arr.get_shape(), shape);

            let arr_ref: &[u32] = &arr;
            assert_eq!(arr_ref.get_shape(), shape);
        }

        {
            let arr: [i32; 3] = [2, 3, 4];
            assert_eq!(arr.get_shape(), shape);

            let arr_ref: &[i32] = &arr;
            assert_eq!(arr_ref.get_shape(), shape);
        }

        {
            let vec: Vec<usize> = vec![2, 3, 4];
            assert_eq!(vec.get_shape(), shape);

            let vec_ref: &Vec<usize> = &vec;
            assert_eq!(vec_ref.get_shape(), shape);
        }

        {
            let vec: Vec<u32> = vec![2, 3, 4];
            assert_eq!(vec.get_shape(), shape);

            let vec_ref: &Vec<u32> = &vec;
            assert_eq!(vec_ref.get_shape(), shape);
        }

        let tensor: Tensor<burn::backend::NdArray, 2> = Tensor::zeros([2, 2], &Default::default());
        let tensor_ref = &tensor;
        assert_eq!(tensor_ref.get_shape(), Shape::from([2, 2]));
    }
}
