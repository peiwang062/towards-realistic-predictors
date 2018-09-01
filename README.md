# towards-realistic-predictors

This repo constains the pytorch implementation for the paper Towards Realistic Predictors on ECCV2018.

## requirements

* python 3.6
* pytorch = 0.4
* other common modules

## Usage

Since our code is not an end-to-end trainable model, please run 
```
train_HPnet_dataset_net_net.py
```
to get the hardness predictor after replacing 'dataset' and 'net' with 'imagenet' or 'indoor' and 'res' or 'vgg'.

Then run 
```
train_rp_dataset_net_net.py
```
to get realistic predictors.

## Contact

For questions, feel free to reach
```
Pei Wang: peiwang062@gmail.com
```
