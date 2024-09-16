#!/bin/bash
#SBATCH --time=3-00:00:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --partition=volta_compute --qos=long
#SBATCH --array=0-15
#SBATCH --output=./%x_%A_%a.out
#SBATCH --error=./%x_%A_%a.err

input_params = ("facebook/xglm-7.5B 2" "facebook/xglm-7.5B 3" "facebook/xglm-7.5B 6" "facebook/xglm-7.5B 7" "facebook/xglm-7.5B 10" "facebook/xglm-7.5B 11" "facebook/xglm-7.5B 14" "facebook/xglm-7.5B 15" "malteos/bloom-6b4-clp-german 2" "malteos/bloom-6b4-clp-german 3" "malteos/bloom-6b4-clp-german 6" "malteos/bloom-6b4-clp-german 7" "malteos/bloom-6b4-clp-german 10" "malteos/bloom-6b4-clp-german 11" "malteos/bloom-6b4-clp-german 14" "malteos/bloom-6b4-clp-german 15")

source /work/home/jsieker/venv/florianIC2/bin/activate
srun --gres=gpu:1 python -u batch_cons_forcedreference_generate.py ${input_params[SLURM_ARRAY_TASK_ID]}