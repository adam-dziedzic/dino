#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --partition=nic
#SBATCH --account=nic
#SBATCH --qos=nic
#SBATCH --mem=120G
#SBATCH -c 10
#SBATCH --nodelist=nic3

source /h/321/ady/.envnew

echo "hostname: ${HOSTNAME}"

export PYTHONPATH="${PYTHONPATH}:/h/321/ady/code/dino"
data_drive=mfsnic
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
echo $timestamp
/h/321/ady/anaconda3/envs/python39/bin/python3.9 -m torch.distributed.launch --nproc_per_node=4 /h/321/ady/code/dino/main_dino.py \
--arch vit_tiny \
--data_path /mfsnic/datasets/imagenet/train \
--output_dir /mfsnic/adam/dino-tiny

# 78565

