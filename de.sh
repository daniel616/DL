#!/bin/bash
#SBATCH -e new_err_%j.txt
#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 10
#SBATCH -o new_o_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.daniel616@gmail.com
#SBATCH --job-name=de



python -u eval_anchor.py
