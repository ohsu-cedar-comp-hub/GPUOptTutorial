#!/bin/bash
#SBATCH --partition gpu
#SBATCH --account CEDAR
#SBATCH --gres=gpu:a40:1    
#SBATCH --array=1-2%2
#SBATCH --cpus-per-task 1
#SBATCH --mem 20G
#SBATCH --time 1:00:00
#SBATCH --job-name tile_run_gpu

source activate base
conda init zsh
conda activate gigapath

python script.py -id TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -hf 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 -lf log/TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -o results/
