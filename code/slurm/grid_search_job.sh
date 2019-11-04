#!/bin/bash

set -eo pipefail

source conda.sh

cd ${HOME}/prog/work/swift.ch-analysis-dws/;

in_file=$1
model=$2
epochs=$3
shuffle=$4
h_reg=$5
l2wt=$6
output_dir=$7
l_rate=$8
training_fraction=1.0

python -u ./code/hlr_learning.py "${in_file}" -m ${model} -epochs ${epochs} -shuffle ${shuffle} -h_reg ${h_reg} -l2wt ${l2wt} -training_fraction ${training_fraction} -o "${output_dir}" -l_rate ${l_rate}
