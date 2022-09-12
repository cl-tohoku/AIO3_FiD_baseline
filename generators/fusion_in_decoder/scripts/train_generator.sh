#!/usr/bin/bash

# if abci
# USAGE: qsub -cwd -g gcb50246 -l rt_F=1 -l h_rt=48:00:00 -N log_fid $HOME/exp2021/FiD/train_generator.sh
# source scripts/abci_setting.sh miniconda3-3.19.0/envs/FiD

python train_generator.py --config_file $1

DATE=`date +%Y%m%d-%H%M`
echo $DATE