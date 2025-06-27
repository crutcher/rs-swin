# Bimm Shape Contracts

* static/stack-evaluated runtime shape contracts for tensors.

## Example Usage

```rust
/// Window Partition
///
/// ## Parameters
///
/// - `tensor`: Input tensor of shape (B, h_wins * window_size, w_wins * window_size, C).
/// - `window_size`: Window size.
///
/// ## Returns
/// 
/// Output tensor of shape (B * h_windows * w_windows, window_size, window_size, C).
/// 
/// ## Panics
/// 
/// Panics if the input tensor does not have 4 dimensions.
pub fn window_partition<B: Backend, K>(
    tensor: Tensor<B, 4, K>,
    window_size: usize,
) -> Tensor<B, 4, K>
where
    K: BasicOps<B>,
{
    static CONTRACT: ShapeContract = ShapeContract::new(&[
        DimMatcher::Expr(DimExpr::Param("batch")),
        DimMatcher::Expr(DimExpr::Prod(&[
            DimExpr::Param("h_wins"),
            DimExpr::Param("window_size"),
        ])),
        DimMatcher::Expr(DimExpr::Prod(&[
            DimExpr::Param("w_wins"),
            DimExpr::Param("window_size"),
        ])),
        DimMatcher::Expr(DimExpr::Param("channels")),
    ]);
    let [b, h_wins, w_wins, c] = CONTRACT.unpack_shape(
        &tensor.dims(),
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)],
    );

    tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c])
}
```

## Performance

Benchmark: `214.11 ns/iter (+/- 4.71)`
```rust
#[bench]
fn bench_shape_contract(b: &mut Bencher) {
    static PATTERN: ShapeContract = ShapeContract::new(&[
        DimMatcher::Any,
        DimMatcher::Expr(DimExpr::Param("b")),
        DimMatcher::Ellipsis,
        DimMatcher::Expr(DimExpr::Prod(&[DimExpr::Param("h"), DimExpr::Param("p")])),
        DimMatcher::Expr(DimExpr::Prod(&[DimExpr::Param("w"), DimExpr::Param("p")])),
        DimMatcher::Expr(DimExpr::Pow(&DimExpr::Param("z"), 3)),
        DimMatcher::Expr(DimExpr::Param("c")),
    ]);

    let batch = 2;
    let height = 3;
    let width = 2;
    let padding = 4;
    let channels = 5;
    let z = 4;

    let shape = [12, batch, 1, 2, 3, height * padding, width * padding, z * z * z, channels];
    let env = [("p", padding), ("c", channels)];
    let keys = ["b", "h", "w", "z"];

    b.iter(|| {
        let _ = PATTERN.unpack_shape(&shape, &keys, &env);
    });
}
```