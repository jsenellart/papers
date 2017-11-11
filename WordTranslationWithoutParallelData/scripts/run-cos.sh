#!/bin/bash

export PYTHONPATH=/home/devling/projects/faiss
export LD_LIBRARY_PATH=/opt/OpenBLAS/lib:/usr/local/cuda-8.0/lib64/

src=$1
tgt=$2
seed=$3

echo "LAUNCH python3.5 -u traincos.py wiki.${src}.vec_200000.bin wiki.${tgt}.vec_200000.bin --vocSize 200000 --gpuid 0 --nEpoch 200 --refinementIterations 6 --dump_output out/dictcos-${src}${tgt}-seed${seed}.txt --save out/Wcos-${src}${tgt}-seed${seed} --seed $seed > out/logcos-${src}${tgt}-seed${seed}.txt"

python3.5 -u traincos.py wiki.${src}.vec_200000.bin wiki.${tgt}.vec_200000.bin --vocSize 200000 --gpuid 0 --nEpoch 200 --refinementIterations 6 --dump_output out/dictcos-${src}${tgt}-seed${seed}.txt --save out/Wcos-${src}${tgt}-seed${seed} --seed $seed > out/logcos-${src}${tgt}-seed${seed}.txt
