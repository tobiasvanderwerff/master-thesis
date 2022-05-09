#!/bin/bash

#SBATCH --job-name='mta'
#SBATCH --partition=gpushort
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#SBATCH --array=1
#--mail-type=FAIL,END
#--mail-user=t.n.van.der.werff@student.rug.nl

set -euo pipefail

module purge
module load Python/3.8.6-GCCcore-10.2.0
source ~/activate_py3.8.6

# pip install -e ../htr
# pip install -e ..

seed=$SLURM_ARRAY_TASK_ID

base_model_arch="fphtr"
main_model_arch="WriterCodeAdaptiveModel"
learning_rate=1e-4
weight_decay=1e-4
shots=8
ways=4
adaptation_opt_steps=1

TRAINED_MODEL_PATH=/data/s4314719/thesis/writer_code/lightning_logs/WriterCodeAdaptiveModel-wer=0.1227=5_lr=1e-4_shots=8_seed=1/checkpoints/WriterCodeAdaptiveModel-epoch=32-char_error_rate=0.0557-word_error_rate=0.0896.ckpt

CACHE_DIR=/data/s4314719/thesis/writer_code/lightning_logs/cache_without_bad_segmentation

experiment_name="WriterCodeAdaptiveModel-wer=0.1227_seed=${seed}_test"

logdir="lightning_logs/${experiment_name}"
logfile="${logdir}/train.out"
srcdir="${logdir}/src"

mkdir -p $logdir
mkdir -p $srcdir

# Copy job file
job_file=$(realpath $0)
cp $job_file $logdir

# Copy source code
cp src/*.py $srcdir

# cd $HOME/master-thesis/writer_code
srun python main.py \
--base_model_arch $base_model_arch \
--main_model_arch $main_model_arch \
--trained_model_path $TRAINED_MODEL_PATH \
--cache_dir $CACHE_DIR \
--experiment_name $experiment_name \
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--num_nodes $SLURM_JOB_NUM_NODES \
--seed $seed \
--shots $shots \
--ways $ways \
--adaptation_opt_steps $adaptation_opt_steps \
--data_dir /data/s4314719/IAM \
--num_workers 12 \
--test \
&> $logfile
