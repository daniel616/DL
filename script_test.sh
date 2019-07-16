#!/bin/bash
#SBATCH -e err-%A.txt
#SBATCH -p gpu-common --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 10
#SBATCH -o results-%A.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.daniel616@gmail.com





#module load Python-GPU/3.6.7
python tools/test.py $1 $2 --show

