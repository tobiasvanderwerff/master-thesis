# stats-per-writer
For a trained meta-learing model, decide whether or not to perform adaptation on the
support batch based on gradient uncertainty. Gradient uncertainty in this
case is expressed as the gradient of the target log probability with respect to the
input image, using model predictions as pseudo targets.

This code is only meant for running val/test loop on a trained model and will crash for
train loop. Additionally, show WER scores both with and without inner loop adaptation.
These scores are saved to a CSV file afterwards.

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
