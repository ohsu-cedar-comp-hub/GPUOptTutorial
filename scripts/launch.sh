#!/bin/bash
#SBATCH --partition gpu
#SBATCH --account CEDAR
#SBATCH --gres=gpu:a40:1  
#SBATCH --array=1-2%2  
#SBATCH --cpus-per-task 1
#SBATCH --mem 20G
#SBATCH --time 1:00:00
#SBATCH --job-name gpu_opt_tut

eval "$(conda shell.bash hook)"
conda init
conda activate /home/exacloud/gscratch/CEDAR/chaoe/miniconda3/envs/gigapath

if [ -n "$1" ]; then
  CACHE_ARG="-c \"$1\""
else
  CACHE_ARG=""
fi


python scripts/script.py -id TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -hf 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 -lf log/TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -o results/ $CACHE_ARG

