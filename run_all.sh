#!/bin/sh

N="50 150 250"
SEG="quickshift felzenszwalb"
REG="linear ridge lasso"
SEL="3"
WEIGHTS="true false"



for n in $N
do
for seg in $SEG
do
for reg in $REG
do
for sel in $SEL
do
for weights in $WEIGHTS
do

echo "python run.py -n $n --seg $seg --reg $reg --sel $sel --weights $weights"
python run.py -n $n --seg $seg --reg $reg --sel $sel --weights $weights

done
done
done
done
done
