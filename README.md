# Burn Image Models

## Overview

This is a Rust crate for image models, inspired by the Python `timm` package.

It is *very* early in development, and is not yet ready for production use.
The Burn ecosystem is still missing a number of image-model related features,
particularly in the area of image datasets and transforms.

## Philosophy

This crate is intended to provide a collection of image models, primarily
translated from the `timm` library, but also including some original models.
It is also intended to provide some of the missing image dataset and transform functionality
that is needed to train these models.

This crate will emphasize a few things not commonly found in the torch ecosystem.

### Composition

The `timm` library contains many models that are monolithic,
despite sharing duplicates of much of the same code.

This crate will focus on providing a set of composable components
for the internal components of the models, so that larger full models,
particularly those in the same family,
can be built from smaller reusable and tested components.

### Shape Contracts

A low-overhead (const, stack-evaluated) shape contract library
that can be used in-line with the models to both ensure tensor geometry,
and unpack complex shape components at runtime; coupled with a high-quality
panic reporting system that provides detailed information about pattern errors.

## Current Status

There is a largely complete implementation of the Swin Transformer V2,
and a training example using the CINIC-10 dataset.

## Contributing

I (crutcher) am relatively new to Rust, and am still learning the idioms of the language;
*particularly* in relation to what makes a good crate.

I believe strongly in tooling (like `clippy` and `rustfmt`),
and 100% test coverage; but I don't have strong knowledge of
how to write idiomatic Rust code yet, particularly in relation to
that tooling.

I'd love help from anyone interested in contributing to this crate.

I generally hang out on the Burn Discord; and development discussions
should probably be held in the `#vision` channel there.

## Setup

Much dev tooling is done with `cargo-make`, so you'll need to install that:

    cargo install cargo-make

Then, you can run the dev setup script to install the necessary dependencies:

    cargo make setup

## Dev Cycle

Run `rustfmt`, `clippy`, and `test':

    cargo make devtest

## Benchmarks

Benchmarks are run by enabling the nightly toolchain:

    cargo +nightly bench --features nightly

Or just:

    cargo make bench
