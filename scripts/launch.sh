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

CACHE=""
while [[ $# -gt 0 ]]; do
    case $1 in
        -c)
            CACHE="$2"
            shift 2
            ;;
        -hf)
            HF_TOKEN="$2"
            shift 2
            ;;
        *)
            echo "Incorrect option: $1"
            exit 1
            ;;
    esac
done

if [[ -z "$HF_TOKEN" ]]; then
    echo "Error: HuggingFace token (-hf) is required."
    exit 1
fi


python scripts/script.py -id TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -hf $HF_TOKEN -lf log/TCGA-BRCA/TCGA-BRCA-batch_${SLURM_ARRAY_TASK_ID} -o results/ -c $CACHE

