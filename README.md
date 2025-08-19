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

This crate provides a set of image-specific operations for `bimm-firehose`.

Add-on crates:
* [bimm-firehose-image](crates/bimm-firehose-image)

## External Related Crates

### [bimm-contracts](https://github.com/crutcher/bimm-contracts) - a crate for static shape contracts for tensors.

[![Crates.io Version](https://img.shields.io/crates/v/bimm-contracts)](https://crates.io/crates/bimm-contracts)
[![docs.rs](https://img.shields.io/docsrs/bimm-contracts)](https://docs.rs/bimm-contracts/latest/bimm-contracts/)

This crate provides a stand-alone library for defining and enforcing tensor shape contracts
in-line with the Burn framework modules and methods.

## Contributing

See the [CONTRIBUTING](CONTRIBUTING.md) guide for build and contribution instructions.
