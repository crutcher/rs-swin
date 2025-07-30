# firehose_loadtest example

This is a basic example of loadtesting the `bimm-firehose` process
over images from the `rs-cinic-10-index` dataset.

It is built upon the `rs-cinic-10-index` dataset, which uses the CINIC-10 dataset.

## Installing the Dataset

See:
* https://github.com/BayesWatch/cinic-10
* https://datashare.ed.ac.uk/handle/10283/3192

1. Download the dataset, and unpack it.
2. Set the environment variable `CINIC10_PATH` to the path of the unpacked dataset.

## Running the Example

Run the training:

`CINIC10_PATH=<path-to-CINIC10> cargo run --release -p firehose_loadtest`

On my dev machine (with a NVMe drive), I see the following timing:
```text
Loaded 90000 images
Total duration: 6.18658221s
batch_size: 512
batch duration: 35.151035ms
item duration: 68.654µs
```

