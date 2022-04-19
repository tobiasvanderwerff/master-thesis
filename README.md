# wang2020

Implementation of CNN adaptation method used in "Writer-Aware CNN for Parsimonious
HMM-Based Offline Handwritten Chinese Text Recognition" by Wang et al. (2020). It is
encapsulated in this image:

![](/home/tobias/Dropbox/master_AI/thesis/code/img/wang2020.png)

Basically the writer code is converted to a channel-wise bias vector by a linear
transformation. The layer output is then of the following form:

![](/home/tobias/Dropbox/master_AI/thesis/code/img/adaptive_output.png)

where M is the regular convolution layer output, and Q is the channel-wise bias
vector derived from the writer code. Note that f is a activation function (ReLU).



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
