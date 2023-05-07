#!/bin/bash
#SBATCH -A research
#SBATCH --qos=medium
#SBATCH -n 19
#SBATCH --gres=gpu:2
#SBATCH --time=1-00:00:00
#SBATCH --output=op_file.txt
#SBATCH --mail-type=END

### activate conda env
source activate pytenv

module load u18/cuda/11.6
module load u18/cudnn/8.4.0-cuda-11.6

echo ---Starting Training---

python cis_train_full.py

echo ----Training Complete----
