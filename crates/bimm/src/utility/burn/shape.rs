use burn::prelude::Shape;
use std::ops::Range;

/// Construct an array of `Range<usize>` covering the `Shape`.
pub fn shape_to_ranges<const D: usize>(shape: Shape) -> [Range<usize>; D] {
    shape
        .dims::<D>()
        .iter()
        .map(|d| 0..*d)
        .collect::<Vec<_>>()
        .try_into()
        .unwrap()
}
