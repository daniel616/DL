#!/bin/bash
#SBATCH -e cascade_err.txt
#SBATCH --mem=16G
#SBATCH -c 10
#SBATCH -o cascade_o.txt
#SBATCH -p dsplus-gpu --gres=gpu:1
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.daniel616@gmail.com
#SBATCH --job-name=cascade




python tools/train.py $1
python tools/test.py $1 $2 --show
