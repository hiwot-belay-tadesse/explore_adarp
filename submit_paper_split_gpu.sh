#!/bin/bash
#SBATCH -J adarp_paper_split
#SBATCH -n 8 # Number of cores (data loading; the CNN runs on the GPU)
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-02:00 # Runtime in D-HH:MM (an A100 should finish the pooled SMOTE model in well under an hour)
#SBATCH -p gpu # GPU partition; gpu_test (12 h, max 2 jobs) is quicker to start for a trial
#SBATCH --gpus=1 # one GPU of any type; e.g. --gpus=nvidia_a100-sxm4-80gb:1 to pin the type
#SBATCH --mem=32GB
#SBATCH -o processed_paper_split/out_%j.txt # File to which STDOUT will be written
#SBATCH -e processed_paper_split/err_%j.txt # File to which STDERR will be written
#
# Usage (from the project directory on the cluster):
#   mkdir -p processed_paper_split
#   interactive test first (per FASRC docs): salloc -p gpu_test -t 0-01:00 --mem 8000 --gpus=1 ; nvidia-smi
#   sbatch submit_paper_split.sh                                  # pooled model, SMOTE (the agreed run)
#   sbatch --export=ALL,ONLY=P101C,BALANCE=under submit_paper_split.sh   # any other (scope, balance)
#   sbatch --export=ALL,ONLY=,BALANCE=under,smote submit_paper_split.sh   # everything: pooled + 11 personal, both balances
# Finished (scope, balance) pairs in results.csv are skipped, so a job can simply be resubmitted after a timeout.

set -eo pipefail                            # any failing step aborts the job, so SLURM reports FAILED instead of COMPLETED
module load python cuda cudnn      # FASRC: load CUDA (and cuDNN) before TensorFlow; pin versions with module spider cuda
source activate ${CONDA_ENV:-adarp} || { echo "conda env ${CONDA_ENV:-adarp} not found: create it with  mamba create -n adarp python=3.11 && source activate adarp && pip install -r requirements_cluster.txt"; exit 1; }

export ONLY=${ONLY-pooled}
export BALANCE=${BALANCE:-smote}
export EPOCHS=${EPOCHS:-50}
export TF_CPP_MIN_LOG_LEVEL=2
nvidia-smi --query-gpu=name,memory.total --format=csv || echo "no GPU visible"

cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname)  ONLY=$ONLY BALANCE=$BALANCE EPOCHS=$EPOCHS  started $(date)"

[ -f processed_paper_split/data.npz ] || python -u prep_paper_split.py
python -u train_paper_split.py
python -u plot_paper_split.py

echo "finished $(date)"
