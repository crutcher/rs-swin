# bimm-contracts

![Crates.io Version](https://img.shields.io/crates/v/bimm-contracts)
[![docs.rs](https://img.shields.io/docsrs/bimm-contracts)](https://docs.rs/bimm-contracts/latest/)

#### Recent Changes

* **0.2.4**
  * no_std support.
* **0.2.3**
  * Extensive documentation.
  * Renamed `ShapeContract::maybe_*` to `ShapeContract::try_*`
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

## Overview

This is a `no_std` inline contract programming library for tensor geometry
for the [burn](https://burn.dev) tensor framework.

Contract programming, or [Design by Contract](https://en.wikipedia.org/wiki/Design_by_contract),
is a programming paradigm that specifies the rights and obligations of software components.

The goal of this library is to make in-line geometry contracts:
* Easy to Read, Write, and Use,
* Performant at Runtime (so they can always be enabled),
* Verbose and Helpful in their error messages.

## API

The primary public API of this library is:
* `shape_contract` - a macro for defining shape contracts.
* `run_every_nth` - a macro for running code on an incrementally lengthening schedule.
* `ShapeContract::assert_shape` - assert a contract.
* `ShapeContract::unpack_shape` - assert a contract, and unpack geometry components.

The shape methods take a `ShapeArgument` parameter; with implementations for:
* ``burn::prelude::Shape``,
* ``&burn::prelude::Shape``,
* ``&burn::prelude::Tensor``,
* ``&[usize]``, ``&[usize; D]``,
* ``&[u32]``, ``&[u32; D]``,
* ``&[i32]``, ``&[i32; D]``,
* ``&Vec<usize>``,
* ``&Vec<u32>``,
* ``&Vec<i32>``,

## Speed and Stack Design

Contracts are only useful when they are fast enough to be always enabled.

As a result, this library is designed to be fast at runtime,
focusing on `static` contracts and using stack over heap wherever possible.

Benchmarks on release builds are available under ``cargo bench -p bimm-contracts``:

```terminaloutput
     Running benches/contracts.rs (target/release/deps/contracts-86950340ff3748c1)
unpack_shape            time:   [176.03 ns 177.39 ns 178.81 ns]
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

assert_shape            time:   [166.57 ns 168.00 ns 169.60 ns]
Found 2 outliers among 100 measurements (2.00%)
  1 (1.00%) high mild
  1 (1.00%) high severe

assert_shape_every_nth/assert_shape_every_nth
                        time:   [4.4057 ns 4.4769 ns 4.5726 ns]
Found 14 outliers among 100 measurements (14.00%)
  6 (6.00%) high mild
  8 (8.00%) high severe
```

## shape_contract! macro

The `shape_contract!` macro is a compile-time macro that parses a shape contract
from a shape contract pattern:

```rust
use bimm_contracts::{ShapeContract, shape_contract};
static CONTRACT: ShapeContract = shape_contract!(_, "x" + "y", ..., "z" ^ 2);
```

A shape pattern is made of one or more dimension matcher terms:
- `_`: for any shape; ignores the size, but requires the dimension to exist.,
- `...`: for ellipsis; matches any number of dimensions, only one ellipsis is allowed,
- a dim expression.

```bnf
ShapeContract => <LabeledExpr> { ',' <LabeledExpr> }* ','?
LabeledExpr => {Param ":"}? <Expr>
Expr => <Term> { <AddOp> <Term> }
Term => <Power> { <MulOp> <Power> }
Power => <Factor> [ ^ <usize> ]
Factor => <Param> | ( '(' <Expression> ')' ) | NegOp <Factor>
Param => '"' <identifier> '"'
identifier => { <alpha> | "_" } { <alphanumeric> | "_" }*
NegOp =>      '+' | '-'
AddOp =>      '+' | '-'
MulOp =>      '*'
```

## Error Messages

Error messages are verbose and helpful.

```rust
use bimm_contracts::{ShapeContract, shape_contract};
use indoc::indoc;

fn example() {
    static CONTRACT: ShapeContract = shape_contract![
            ...,
            "hwins" * "window",
            "wwins" * "window",
            "color",
        ];

    let hwins = 2;
    let wwins = 3;
    let window = 4;
    let color = 3;

    let shape = [1, 2, 3, hwins * window, wwins * window, color];

    let [h, w] = CONTRACT.unpack_shape(&shape, &["hwins", "wwins"], &[
        ("window", window),
        ("color", color),
    ]);
    assert_eq!(h, hwins);
    assert_eq!(w, wwins);

    assert_eq!(
        CONTRACT.try_unpack_shape(&shape, &["hwins", "wwins"], &[
            ("window", window + 1),
            ("color", color),
        ]).unwrap_err(),
        indoc! {r#"
            Shape Error:: 8 !~ (hwins*window) :: No integer solution.
             shape:
              [1, 2, 3, 8, 12, 3]
             expected:
              [..., (hwins*window), (wwins*window), color]
              {"window": 5, "color": 3}"#
        },
    );
}
```

## Usage Example

```rust
use burn::prelude::{Tensor, Backend};
use burn::tensor::BasicOps;
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
    // In release builds, this has an benchmark of ~170ns:
    let [b, h_wins, w_wins, c] = INPUT_CONTRACT.unpack_shape(
        &tensor,
        &["batch", "h_wins", "w_wins", "channels"],
        &[("window_size", window_size)],
    );

    let tensor = tensor
        .reshape([b, h_wins, window_size, w_wins, window_size, c])
        .swap_dims(2, 3)
        .reshape([b * h_wins * w_wins, window_size, window_size, c]);

    // Run an amortized check on the output shape.
    //
    // `run_every_nth!{}` runs the first 10 times,
    // then on an incrementally lengthening schedule,
    // until it reaches its default period of 1000.
    //
    // Due to amortization, in release builds, this averages ~4ns:
    run_every_nth!({
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
