#[cfg(feature = "burn_support")]
use burn::prelude::{Backend, Shape, Tensor};
#[cfg(feature = "burn_support")]
use burn::tensor::BasicOps;

/// A trait that provides a method to extract the shape from various types.
pub trait ShapeArgument {
    /// Extracts the shape from the implementing type as a vector.
    fn get_shape_vec(self) -> Vec<usize>;
}

#[cfg(feature = "burn_support")]
impl ShapeArgument for &Shape {
    fn get_shape_vec(self) -> Vec<usize> {
        self.dims.clone()
    }
}

#[cfg(feature = "burn_support")]
impl ShapeArgument for Shape {
    fn get_shape_vec(self) -> Vec<usize> {
        self.dims
    }
}

#[cfg(feature = "burn_support")]
impl<B, const D: usize, K> ShapeArgument for &Tensor<B, D, K>
where
    B: Backend,
    K: BasicOps<B>,
{
    fn get_shape_vec(self) -> Vec<usize> {
        self.shape().dims.clone()
    }
}

impl<const D: usize> ShapeArgument for &[usize; D] {
    fn get_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl<const D: usize> ShapeArgument for &[u32; D] {
    fn get_shape_vec(self) -> Vec<usize> {
        self.iter().map(|&d| d as usize).collect::<Vec<_>>()
    }
}

impl<const D: usize> ShapeArgument for &[i32; D] {
    fn get_shape_vec(self) -> Vec<usize> {
        self.iter().map(|&d| d as usize).collect::<Vec<_>>()
    }
}

impl ShapeArgument for &[usize] {
    fn get_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl ShapeArgument for &[u32] {
    fn get_shape_vec(self) -> Vec<usize> {
        self.iter().map(|&d| d as usize).collect::<Vec<_>>()
    }
}

impl ShapeArgument for &[i32] {
    fn get_shape_vec(self) -> Vec<usize> {
        self.iter().map(|&d| d as usize).collect::<Vec<_>>()
    }
}

impl ShapeArgument for &Vec<usize> {
    fn get_shape_vec(self) -> Vec<usize> {
        self.to_vec()
    }
}

impl ShapeArgument for &Vec<u32> {
    fn get_shape_vec(self) -> Vec<usize> {
        self.iter().map(|&d| d as usize).collect::<Vec<_>>()
    }
}

impl ShapeArgument for &Vec<i32> {
    fn get_shape_vec(self) -> Vec<usize> {
        self.iter().map(|&d| d as usize).collect::<Vec<_>>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_argument() {
        let expected = vec![2, 3, 4];

        {
            let arr: [usize; 3] = [2, 3, 4];
            assert_eq!(&arr.get_shape_vec(), &expected);

            let arr_ref: &[usize] = &arr;
            assert_eq!(&arr_ref.get_shape_vec(), &expected);
        }

        {
            let arr: [u32; 3] = [2, 3, 4];
            assert_eq!(&arr.get_shape_vec(), &expected);

            let arr_ref: &[u32] = &arr;
            assert_eq!(&arr_ref.get_shape_vec(), &expected);
        }

        {
            let arr: [i32; 3] = [2, 3, 4];
            assert_eq!(&arr.get_shape_vec(), &expected);

            let arr_ref: &[i32] = &arr;
            assert_eq!(&arr_ref.get_shape_vec(), &expected);
        }

        {
            let vec: Vec<usize> = vec![2, 3, 4];
            assert_eq!(&vec.get_shape_vec(), &expected);

            let vec_ref: &Vec<usize> = &vec;
            assert_eq!(&vec_ref.get_shape_vec(), &expected);
        }

        {
            let vec: Vec<u32> = vec![2, 3, 4];
            assert_eq!(&vec.get_shape_vec(), &expected);

            let vec_ref: &Vec<u32> = &vec;
            assert_eq!(&vec_ref.get_shape_vec(), &expected);
        }

        #[cfg(feature = "burn_support")]
        {
            let shape = Shape::from([2, 3, 4]);
            assert_eq!(&shape.clone().get_shape_vec(), &expected);

            let shape_ref: &Shape = &shape;
            assert_eq!(&shape_ref.get_shape_vec(), &expected);

            let tensor: Tensor<burn::backend::NdArray, 2> =
                Tensor::zeros([2, 2], &Default::default());
            let tensor_ref = &tensor;
            assert_eq!(&tensor_ref.get_shape_vec(), &[2, 2]);
        }
    }
}
