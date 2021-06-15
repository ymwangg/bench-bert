#!/bin/bash
for i in `cat models.txt`
do
    bash download_model.sh $i
done
