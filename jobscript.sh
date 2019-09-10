#!/bin/bash
#SBATCH --mem=16G
#SBATCH -c 10
#SBATCH -o mask%A.txt
#SBATCH -p dsplus-gpu
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.daniel616@gmail.com
#SBATCH --job-name=mask



python tools/train.py $1
python tools/test.py $1 $2 --show
