# ada-bn-stats
Conditional batch normalization where the input codes are derived from layer
statistics (mean and variance) per channel. Concretely, the L2-distance is
measured between the overall mean and variance for a particular layer and the
writer-specific mean and variance. E.g., for a batchnorm layer with 64 input
channels, the input vector to conditional batchnorm would be 64*2-dimensional.

# Master thesis

## How to install
```shell
git@github.com:tobiasvanderwerff/master-thesis.git  # uses SSH
cd master-thesis
git submodule update --init
pip install -e htr
pip install -e .
```

## Points of attention
- 16-bit mixed precision cannot be used right now in combination with the
  `learn2learn` lib. This is because the `learn2learn` lib calls backpropagation
  for you when calling the inner loop adaptation function. This means the Pytorch
  Lightning cannot scale the gradients etc. accordingly when doing backpropagation.
