#!/usr/bin/env bash

set -e

# for h_reg in {0.1,0.01,0.001,0.0001,0.00001,0.000001}
# do
#   for l2wt in {1.0,0.1,0.01,0.001};
#   do
#     python experiment.py swift_duo.full.csv -m power -epochs 25 -shuffle 41 -h_reg ${h_reg} -l2wt ${l2wt} &
#     echo $!
#   done
# done


# power:
model=power
h_reg=0.000100
l2wt=0.1

# # hlr:
# model=hlr
# h_reg=0.00001
# l2wt=0.01

# for training_fraction in {0.1,0.3,0.5,0.7,0.9,1.0};
# do
training_fraction=1.0
for model in {"hlr","power"}
do
  for h_reg in {0.1,0.01,0.001,0.0001,0.00001,0.000001}
  do
    for l2wt in {1.0,0.1,0.01,0.001};
    do
      ./code/hlr_learning.py ./processed/swift_duo.csv -m ${model} -epochs 25 -shuffle 41 -h_reg ${h_reg} -l2wt ${l2wt} -training_fraction ${training_fraction} -o ./processed/hlr/ &
      echo $!;
    done
  done
done


## Run as:
## ./pool_man_parallel.sh > pids
## Then kill recalcitrant processes from `pids` file, if needed.
