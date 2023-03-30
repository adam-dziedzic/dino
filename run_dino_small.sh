#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --partition=nic
#SBATCH --account=nic
#SBATCH --qos=nic
#SBATCH --mem=500G
#SBATCH -c 10
#SBATCH --nodelist=boorchu

source /h/321/ady/.envnew

echo $HOSTNAME

export PYTHONPATH="${PYTHONPATH}:/h/321/ady/code/dino"
data_drive=mfsnic
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
echo $timestamp
/h/321/ady/anaconda3/envs/python39/bin/python3.9 -m torch.distributed.launch --nproc_per_node=4 /h/321/ady/code/dino/main_dino.py \
--arch vit_small \
--data_path /mfsnic/datasets/imagenet/train \
--output_dir /mfsnic/adam/dino

# 78562

