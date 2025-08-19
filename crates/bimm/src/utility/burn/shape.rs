use burn::prelude::Shape;
use std::ops::Range;

/// Construct an array of `Range<usize>` covering the [`Shape`].
pub fn shape_to_ranges<const D: usize>(shape: Shape) -> [Range<usize>; D] {
    shape
        .dims::<D>()
        .iter()
        .map(|d| 0..*d)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_shape_to_ranges() {
        let shape = Shape::new([2, 3, 4]);
        let ranges = shape_to_ranges::<3>(shape);
        assert_eq!(ranges, [0..2, 0..3, 0..4]);
    }
}
