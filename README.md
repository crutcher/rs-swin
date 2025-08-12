# Burn Image Models

[![Coverage Status](https://coveralls.io/repos/github/crutcher/bimm/badge.svg?branch=main)](https://coveralls.io/github/crutcher/bimm?branch=main)

## Overview

This is a Rust crate for image models, inspired by the Python `timm` package.

It is *very* early in development, and is not yet ready for production use.
The Burn ecosystem is still missing a number of image-model related features,
particularly in the area of image datasets and transforms.

## Active Surface

Image models require complex data loading and augmentation pipelines,
and the current active surface is focused on building a SQL-inspired
table + operations framework for modular data pipeline construction.

This is being done in the [bimm-firehose](crates/bimm-firehose) crate,
which will be a general-purpose data loading and augmentation framework,
but is not yet ready for use.

A functional demo of `firehose` integration is available in the
[swin_tiny](examples/swin_tiny/src/main.rs) example.

## Crates

### [bimm](crates/bimm) - the main crate for image models.

[![Crates.io Version](https://img.shields.io/crates/v/bimm)](https://crates.io/crates/bimm)
[![docs.rs](https://img.shields.io/docsrs/bimm)](https://docs.rs/bimm/latest/bimm/)

This crate provides a collection of image models, and their constituent sub-components.

The goal is to incrementally clone `timm`'s coverage of the SOTA image models,
while focusing on decomposing the models into reusable, fully tested components.

Currently, this has `SWIN Transformer V2`.

The current work surface is focused on extending the dataloader framework to support
more flexible image datasets and transforms, and the kind of composable image augmentation
that is common in the torch ecosystem library.

### [bimm-firehose](crates/bimm-firehose) - a data loading and augmentation framework.

[![Crates.io Version](https://img.shields.io/crates/v/bimm-firehose)](https://crates.io/crates/bimm-firehose)
[![docs.rs](https://img.shields.io/docsrs/bimm-firehose)](https://docs.rs/bimm/latest/bimm-firehose/)

This crate provides a SQL-inspired table + operations framework for modular data pipeline construction.

It's still very much a work in progress, and any issues/design bugs reported
are very appreciated.

### [bimm-contracts](crates/bimm-contracts) - a crate for static shape contracts for tensors.

[![Crates.io Version](https://img.shields.io/crates/v/bimm-contracts)](https://crates.io/crates/bimm-contracts)
[![docs.rs](https://img.shields.io/docsrs/bimm-contracts)](https://docs.rs/bimm-contracts/latest/bimm-contracts/)

This crate provides a stand-alone library for defining and enforcing tensor shape contracts
in-line with the Burn framework modules and methods.

This includes:
- A macro for defining shape contracts.
- static shape contracts.
- stack-checked (minimizing heap usage) shape assertions.
- an interface for unpacking tensor shapes into their components,
  allowing for parameterized dimensions; and nice error messages
  when the shape does not match the contract.
- a macro for running shape checks periodically,
  amortizing the cost of checks over multiple calls.

```rust
use bimm_contracts::{ShapeContract, shape_contract, run_every_nth};

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


## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) guide for build and contribution instructions.
