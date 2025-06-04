#!/bin/bash
#SBATCH --partition gpu
#SBATCH --account CEDAR
#SBATCH --gres=gpu:a40:1    
#SBATCH --cpus-per-task 1
#SBATCH --mem 20G
#SBATCH --time 4:00:00
#SBATCH --job-name tile_test_crop

source activate base
conda init zsh
conda activate gigapath

python scripts/mini_script_error.py -id TCGA-BRCA/TCGA-BRCA-batch_test -hf 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 -lf log/TCGA-BRCA/TCGA-BRCA-test -o results/

