#!/bin/bash

# Hyperparameters
# ***********************************
feature="hinge"
binarize=1  # boolean, either 0 or 1
sfx=".png"

# imgpath="./img"
outpath="./$feature"
# ***********************************

imgpath=$1
rootpath=$(pwd)

# Read the image path from stdin, i.e. piped input
# read imgpath

if ! [[ -d $imgpath ]]; then
    echo "Invalid image path: $imgpath"
    echo "Exiting."
    exit 1
fi

echo "Extracting $feature features."

case $feature in
    "hinge")
        feature_code=0
        ;;
    "quadhinge")
        feature_code=1
        ;;
    "cohinge")
        feature_code=2
        ;;
    "cochaincode-hinge")
        feature_code=3
        ;;
    "triplechaincode-hinge")
        feature_code=4
        ;;
    "delta-hinge")
        feature_code=5
        ;;
    *)
        echo "Invalid feature: $feature"
        exit 1
        ;;
esac

mkdir -p $outpath

# Convert images to pgm (required for the Hinge binary)
echo "Converting images to pgm."
cd $imgpath
mogrify -format pgm "*$sfx"
cd $rootpath

# Run through all the images
echo "Running feature extraction."
for img in $(ls $imgpath/*.pgm)
do
	id=$(basename $img .pgm)
	echo "Processing $id"
	sudo ./beyondOCR_hingeV13B31 $img $id "$outpath/${id}.hinge" $feature_code $binarize
done

# Delete unneeded produced files
echo "Deleting pgm and ppm files."
rm $imgpath/*.pgm
rm binary.ppm

echo "Done. Features stored in ${outpath}."
