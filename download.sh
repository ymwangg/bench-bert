#!/bin/bash
for i in `cat models.txt`
do
    bash conv.sh $i
done
