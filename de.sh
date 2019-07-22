#!/bin/bash
#SBATCH -e de.txt
#SBATCH -p dsplus-gpu --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 10
#SBATCH -o de.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.daniel616@gmail.com
#SBATCH --job-name=de




python eval_anchor.py
