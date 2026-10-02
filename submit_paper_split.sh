#!/bin/bash
#SBATCH -J adarp_paper_split
#SBATCH -n 16 # Number of cores (TensorFlow uses them all for the CNN)
#SBATCH -N 1 # Ensure that all cores are on one machine
#SBATCH -t 0-12:00 # Runtime in D-HH:MM (pooled SMOTE took ~3 min/epoch on 10 laptop cores)
#SBATCH -p shared # odyssey partition (CPU only; serial_requeue is faster to start but can be preempted mid-model)
#SBATCH --mem=32GB # Memory pool for all cores (see also --mem-per-cpu)
#SBATCH -o processed_paper_split/out_%j.txt # File to which STDOUT will be written
#SBATCH -e processed_paper_split/err_%j.txt # File to which STDERR will be written
#
# Usage (from the project directory on the cluster):
#   mkdir -p processed_paper_split
#   sbatch submit_paper_split.sh                                  # pooled model, SMOTE (the agreed run)
#   sbatch --export=ALL,ONLY=P101C,BALANCE=under submit_paper_split.sh   # any other (scope, balance)
#   sbatch --export=ALL,ONLY=,BALANCE=under,smote submit_paper_split.sh   # everything: pooled + 11 personal, both balances
# Finished (scope, balance) pairs in results.csv are skipped, so a job can simply be resubmitted after a timeout.

set -eo pipefail                            # any failing step aborts the job, so SLURM reports FAILED instead of COMPLETED
module load python
source activate ${CONDA_ENV:-adarp} || { echo "conda env ${CONDA_ENV:-adarp} not found: create it with  mamba create -n adarp python=3.11 && source activate adarp && pip install -r requirements_cluster.txt"; exit 1; }

export ONLY=${ONLY-pooled}
export BALANCE=${BALANCE:-smote}
export EPOCHS=${EPOCHS:-50}
export TF_CPP_MIN_LOG_LEVEL=2
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE TF_NUM_INTRAOP_THREADS=$SLURM_CPUS_ON_NODE TF_NUM_INTEROP_THREADS=2

cd "$SLURM_SUBMIT_DIR"
echo "host $(hostname)  ONLY=$ONLY BALANCE=$BALANCE EPOCHS=$EPOCHS  started $(date)"

[ -f processed_paper_split/data.npz ] || python -u prep_paper_split.py
python -u train_paper_split.py
python -u plot_paper_split.py

echo "finished $(date)"
