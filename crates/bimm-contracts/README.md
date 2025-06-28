# bimm-contracts

![Crates.io Version](https://img.shields.io/crates/v/bimm-contracts)
[![docs.rs](https://img.shields.io/docsrs/bimm-contracts)](https://docs.rs/bimm-contracts/latest/)

Shape Contracts for Burn Image Models (BIMM).

* static/stack-evaluated runtime shape contracts for tensors.

## Changelog

- **0.1.7**: Removed `assert_shape_every_n` in favor of `run_every_nth!` macro.
- **0.1.6**: Added `run_every_nth!` macro.

## Example Usage

```rust
use bimm_contracts::{ShapeContract, DimMatcher, DimExpr, run_every_nth};

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
    static INPUT_CONTRACT: ShapeContract = ShapeContract::new(&[
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
    let [b, h_wins, w_wins, c] = INPUT_CONTRACT.unpack_shape(
        &tensor,
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)],
    );

    let tensor = tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c]);

    // I'd normally not use a contract here, as the shape is already
    // very clear from the above operations; but this is an example
    // of low-overhead periodic shape checking.
    static OUTPUT_CONTRACT: ShapeContract = ShapeContract::new(&[
        DimMatcher::Expr(DimExpr::Prod(&[
            DimExpr::Param("batch"),
            DimExpr::Param("h_wins"),
            DimExpr::Param("w_wins"),
        ])),
        DimMatcher::Expr(DimExpr::Param("window_size")),
        DimMatcher::Expr(DimExpr::Param("window_size")),
        DimMatcher::Expr(DimExpr::Param("channels")),
    ]);
    // Run this check periodically on a growing schedule,
    // up to the default of every 1000th call.
    run_every_nth!(OUTPUT_CONTRACT.assert_shape(
        &tensor,
        &[
            ("batch", b),
            ("h_wins", h_wins),
            ("w_wins", w_wins),
            ("window_size", window_size),
            ("channels", c),
        ]
    ));

    tensor
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

## run_every_nth!(CONTRACT.assert_shape(&tensor, &env))

Benchmark: `4.37 ns/iter (+/- 0.08)`

```rust
#[bench]
fn bench_run_every_nth_assert_shape(b: &mut Bencher) {
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

    let shape = [
        12,
        batch,
        1,
        2,
        3,
        height * padding,
        width * padding,
        z * z * z,
        channels,
    ];
    let env = [("p", padding), ("c", channels)];

    b.iter(|| {
        run_every_nth!(PATTERN.assert_shape(&shape, &env));
    });
}
```