#!/bin/bash
set -ex

cargo install -q cargo-tarpaulin
cargo install -q cargo-make
cargo install -q cargo-criterion
cargo install -q cargo-release

