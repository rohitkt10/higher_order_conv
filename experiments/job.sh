#!/bin/bash
 
# UGE options:
# Run from current working directory
#$ -cwd
# qstat and qacct show maxvmem 55GB in earlier run:
#$ -l m_mem_free=54G
# This is a single GPU example:
#$ -l gpu=1

# show loaded software for this run; will go to stderr
module list

# run program 
python train_encode_residual_bind_nn.py --tfid $1
