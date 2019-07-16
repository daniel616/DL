#!/bin/bash
#SBATCH -e err-%A.txt
#SBATCH -p dsplus-gpu --gres=gpu:1
#SBATCH --mem=16G
#SBATCH -c 10
#SBATCH -o results-%A.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=li.daniel616@gmail.com





#module load Python-GPU/3.6.7
#python tools/train.py $1 #--resume_from #work_dirs/retinanet_r101_fpn_1x/epoch_12.pth
python tools/train.py configs/dan/srs/cascade_rcnn_r101_fpn_1x.py --resume_from work_dirs/cascade_rcnn_r101_fpn_1x/latest.pth
