# towards-realistic-predictors

This repo constains the pytorch implementation for the paper Towards Realistic Predictors on ECCV2018.

## requirements

* python 3.6
* pytorch = 0.4
* other common modules

## Usage

Since our code is not an end-to-end trainable model (for a improved new version, please turn to [an end to end for realistic predictors](https://github.com/peiwang062/end2end_realistic_predictors), please run 
```
train_HPnet_dataset_net_net.py
```
to get the hardness predictor after replacing 'dataset' and 'net' with 'imagenet' or 'indoor' and 'res' or 'vgg'.

Then run 
```
train_rp_dataset_net_net.py
```
to get realistic predictors.

We used the custom data load form, so before training, please first make a train and test sample list. Each item is formed as 
```
imagepath target index
```


## Contact

For questions, feel free to reach
```
Pei Wang: peiwang062@gmail.com
```
