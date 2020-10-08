# wavenet-time-series-forecasting
Borovykn et al. adapted DeepMind's WaveNet for time series forecasting, achieving superb results on many time series tasks. 

This is my implementation of their model in Pytorch, built inside a custom model API. The network captures autocorrelations and correlations with related time series using 1-dimensional dialted causal convolutions, which allows the model to see significantly long lags without blowing up in its number of parameters. 

Skip-connections in each dilated causal convolution stack similar to those found in ResNet allows the network to easily learn identity mappings by driving some weights to zero, allowing multiple layers to be stacked while avoiding getting stuck at sub-optimal local minima.

Rolling origin evaluation is implemented to de-bias out-of-sample forecast errors and hyperparameter optimization is performed using HyperOpt.

Borovykh et al. 2018
https://arxiv.org/pdf/1703.04691.pdf

Bergstra et al. 2011
https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf
