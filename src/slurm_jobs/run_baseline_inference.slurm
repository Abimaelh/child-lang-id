#!/bin/bash

#SBATCH --job-name=child_langid_baseline
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=abimaelh@uw.edu

#SBATCH --account=stf
#SBATCH --partition=gpu-2080ti
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --mem=32G
#SBATCH --gpus=1
#SBATCH --time=01:00:00

#SBATCH --chdir=/mmfs1/home/abimaelh/child-lang-id
#SBATCH --export=all
#SBATCH --output=/mmfs1/home/abimaelh/logs/%x_%j.out
#SBATCH --error=/mmfs1/home/abimaelh/logs/%x_%j.err

source ~/.bashrc
conda activate speechbrain-env

export DATA_DIR=/gscratch/stf/abimaelh/Shiro_Corpus_Segments
export MODEL_DIR=/gscratch/stf/abimaelh/speechbrain_models

python baseline_inference.py
