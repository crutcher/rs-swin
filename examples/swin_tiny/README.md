# swin_tiny example

This example shows how to use a Swin Transformer V2 Tiny model for image classification.

It is built upon the `rs-cinic-10-index` dataset, which uses the CINIC-10 dataset.

It is a goal to move to `burn`s `ImageFolderDataset` in the future, but for now it uses a custom dataset implementation.
`burn`s machinery does not support image-color coercion yet, which is required for the CINIC-10 dataset
because it contains images in sevarl color formats.

## Installing the Dataset

See:
* https://github.com/BayesWatch/cinic-10
* https://datashare.ed.ac.uk/handle/10283/3192

1. Download the dataset, and unpack it.
2. Set the environment variable `CINIC10_PATH` to the path of the unpacked dataset.

## Running the Example

Run the training:

`cargo run --release -p swin_tiny`


