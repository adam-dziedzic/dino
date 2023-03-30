#!/bin/bash

#SBATCH --gres=gpu:4
#SBATCH --partition=nic
#SBATCH --account=nic
#SBATCH --qos=nic
#SBATCH --mem=120G
#SBATCH -c 10
#SBATCH --nodelist=nic2

source /h/321/ady/.envnew

echo $HOSTNAME

export PYTHONPATH="${PYTHONPATH}:/h/321/ady/code/dino"
data_drive=mfsnic
timestamp=$(date +%Y-%m-%d-%H-%M-%S-%N)
echo $timestamp
/h/321/ady/anaconda3/envs/python39/bin/python3.9 -m torch.distributed.launch --nproc_per_node=4 /h/321/ady/code/dino/eval_linear.py \
--arch vit_small \
--data_path /mfsnic/datasets/imagenet \
--output_dir /mfsnic/adam/dino-small-fine-tune

# 78562

