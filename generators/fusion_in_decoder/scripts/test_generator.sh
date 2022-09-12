#!/usr/bin/bash

# if abci
# qsub -cwd -g gcb50246 -l rt_G.small=1 -l h_rt=1:00:00 -N log_fid_test $HOME/exp2021/FiD/test_reader.sh
# source scripts/abci_setting.sh miniconda3-3.19.0/envs/FiD

python test_generator.py --config_file $1 

DATE=`date +%Y%m%d-%H%M`
echo $DATE
