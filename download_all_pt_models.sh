#!/bin/bash
if [ -d "models" ]; then
    rm -rf ./models
fi
mkdir models
for i in `cat models.txt`
do
    ./download_pt_model.sh $i
done
