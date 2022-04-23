# hinge-code
Using traditional Hinge features as writer code. This does not require episodice
training, since the writer codes are ready in advance and don't require
gradient-based training. Right now the procedure goes as follows:

```
for each writer do
    get line images for a single form
    concatenate the lines into a single image
    extract Hinge features for the image
    normalize the Hinge features, which act as the writer code
end for
```

## Currently implemented features
* Hinge. Size: 465
* Quadhinge. Size: 5184
* Cohinge. Size: 10000
* Cochaincode-hinge. Size: 64
* Triplechaincode-hinge. Size: 512
* Delta-hinge. Size: 780


## Example of how to extract features
```shell
# Note: make sure there is about 4GB of space free in the temporary directory. The pgm
# file format takes up a lot of space. Also make sure the Hinge binary (called
# `beyondOCR_hingeV13B31`) is in the `hinge-feature-extraction` folder.

cd hinge-feature-extraction
tempdir="$(pwd)/tmp"  # make a temporary directory for storing images
mkdir -p $tempdir/{img,img-concat}
cp -r ~/datasets/IAM/lines/*/* $tempdir/img  # copy all line images per form

# These two commands can take some time to run. If you want to use a different feature
# (e.g. QuadHinge), specify this inside the `extract-hinge-features.sh` script.
python concat_lines --img_dir $tempdir/img --out_dir $tempdir/img-concat  # concatenate lines
 ./extract-hinge-features.sh $tempdir/img-concat  # extract features

# Remove the temporary directory
rm -r $tempdir
```

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
