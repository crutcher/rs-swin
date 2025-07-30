# bimm-contracts

![Crates.io Version](https://img.shields.io/crates/v/bimm-contracts)
[![docs.rs](https://img.shields.io/docsrs/bimm-contracts)](https://docs.rs/bimm-contracts/latest/)

#### Recent Changes

* **0.2.0**
  * bumped `burn` dependency to `0.18.0`.
* **0.1.9**
   * Improved docs and examples.
   * Internal refactoring for labeled DimMatchers (not supported by macro yet).
* **0.1.8**
   * Added `shape_contract!` macro for easier contract definition.
* **0.1.7**
   * Removed `assert_shape_every_n` in favor of `run_every_nth!` macro.
   * Improved isolation of `run_every_nth!`.

## Benchmarks

[benchmarks](src/benchmarks.rs) are provided to measure the performance of the contract checks.
```
test benchmarks::bench_assert_shape           ... bench:         159.44 ns/iter (+/- 4.23)
test benchmarks::bench_assert_shape_every_nth ... bench:           4.36 ns/iter (+/- 0.03)
test benchmarks::bench_unpack_shape           ... bench:         175.21 ns/iter (+/- 3.99)
```

`assert_shape()` checks are very slightly faster than `unpack_shape()`;
and amortized checks are extremely fast, on average.

## Overview

`bimm-contracts` is runtime tensor-api contract enforcement library for the `burn` machine learning framework;
developed as an independently installable crate of the [Burn Image Models](https://github.com/crutcher/bimm) project.

Contract programming, or [design by contract](https://en.wikipedia.org/wiki/Design_by_contract),
is a programming paradigm that specifies the rights and obligations of software components.

This crate provides a set of tools to define and enforce shape contracts for tensors in the Burn framework,
aiming for:

- *easy authorship* of contracts,
- *static allocation* of contracts,
- *low-heap overhead* for runtime checks,
- *fast fast-path execution*,
- *informative error messages*,
- *amortization of checks* over multiple calls.

## Api

`bimm-contracts` is made of a handful of primary API components:

1. **shape_contract!**: The primary construction interface, a macro to simplify defining a `ShapeContract`.
2. **ShapeContract**: A contract that defines the expected shape of a tensor.
   - **ShapeContract::unpack_shape()**: Unpacks a tensor shape into its components, allowing for parameterized dimensions.
   - **ShapeContract::assert_shape()**: Asserts that a tensor matches the expected shape defined by the contract.
3. **DimMatcher, DimExpr**: A structural pattern language, built of statically allocated pattern components:
   - **DimMatcher**: Represents a single dimension matcher, which can be a parameter, an expression, or a wildcard.
   - **DimExpr**: Represents an expression that can be used to define complex dimension relationships.
4. **run_every_nth!**: A macro that allows for periodic shape checks, amortizing the cost of checks over multiple calls.

## ShapeContracts

A `ShapeContract` is a sequence of `DimMatcher`s that defines the expected shape of a tensor. The matchers are:

- **DimMatcher::Any**: Matches a single dimension, of any size.
- **DimMatcher::Ellipsis**: Matches zero or more dimensions, allowing for flexible shapes.
- **DimMatcher::Expr(DimExpr)**: Matches a dimension based on an expression,
  which can include parameters and operations like multiplication, addition, and exponentiation.

We'll generally use a short-hand syntax for defining `ShapeContract`s, which is a sequence of `DimMatcher`s,
where each matcher can be a parameter, an expression, or a wildcard.

Consider the contract, using the `shape_contract!` macro:

```rust
    static CONTRACT : ShapeContract =
        shape_contract!["batch", ..., "h_wins" * "window_size", "w_wins" * "window_size", "channels"];
```

1. This matches shapes of 4 or more dimensions.
2. The first dimension is a parameter named `"batch"`.
3. Any (non-negative) number of dimensions can follow, represented by `...`.
4. The next two dimensions are products of parameters:
   - The third dimension is the product of `"h_wins"` and `"window_size"`.
   - The fourth dimension is the product of `"w_wins"` and `"window_size"`.
5. The last dimension is a parameter named `"channels"`.

This could also have been constructed manually using the `ShapeContract` constructor:

```rust
static CONTRACT : ShapeContract = ShapeContract::new(&[
    DimMatcher::Expr(DimExpr::Param("batch")),
    DimMatcher::Ellipsis,
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
```

## Dim Expressions

The `DimExpr` expression language permits algebraic expressions over dimensions,
provided that, at assert and/or unpack time, at most one of the parameters is unknown.

The operations supported are limited by those for which a simple solver can be implemented,
and the currently supported operations are:

- **Param**: A named parameter, pulled from the binding environment.
- **A + B**: Addition.
- **A - B**: Subtraction.
- **A * B**: Multiplication.
- **A ^ N**: Exponentiation, where the exponent is a positive constant.
- **A + (B * C)**: Mixed and grouped operations.

## Example Usage

```rust
use bimm_contracts::{ShapeContract, shape_contract, run_every_nth};

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
    static INPUT_CONTRACT: ShapeContract = shape_contract![
        "batch",
        "h_wins" * "window_size",
        "w_wins" * "window_size",
        "channels"
    ];
    let [b, h_wins, w_wins, c] = INPUT_CONTRACT.unpack_shape(
        &tensor,
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)],
    );

    let tensor = tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c]);

    // Run this check periodically on a doubling schedule,
    // up to the default of every 1000th call.
    run_every_nth!({
        // I'd normally not use a contract here, as the shape is already
        // very clear from the above operations; but this is an example
        // of low-overhead periodic shape checking.
        static OUTPUT_CONTRACT: ShapeContract = shape_contract![
            "batch" * "h_wins" * "w_wins",
            "window_size",
            "window_size",
            "channels"
        ];
        OUTPUT_CONTRACT.assert_shape(
            &tensor,
            &[
                ("batch", b),
                ("h_wins", h_wins),
                ("w_wins", w_wins),
                ("window_size", window_size),
                ("channels", c),
            ]
        );
    });
    
    tensor
}
```

