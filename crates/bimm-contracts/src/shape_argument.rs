//! # Utility crate for [`ShapeArgument`] for passing shapes in a type-safe manner.

use alloc::vec::Vec;

/// A trait that provides a method to extract the shape from various types.
pub trait ShapeArgument {
    /// Extracts the shape from the implementing type as a vector.
    fn get_shape_vec(self) -> Vec<usize>;
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
    use alloc::vec;
    use alloc::vec::Vec;

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
    }
}
