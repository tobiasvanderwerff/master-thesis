#!/bin/bash

#SBATCH --job-name='mta'
#SBATCH --partition=gpushort
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:v100
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --mem=16GB
#SBATCH --output=/dev/null
#--mail-type=FAIL,END
#--mail-user=t.n.van.der.werff@student.rug.nl
#SBATCH --array=1

set -euo pipefail

module purge
module load Python/3.9.5-GCCcore-10.3.0
source /data/s4314719/thesis-exp/env/bin/activate

use_msgd="no"
base_model_arch="fphtr"
enc_version=18
learning_rate=1e-4
inner_lr=1e-4
initial_inner_lr=1e-4
weight_decay=0
shots=16
# shots=8
ways=8
max_val_batch_size=128
num_inner_steps=1
max_epochs=-1
# max_epochs=20
early_stopping_patience=5
check_val_every_n_epoch=1
seed=$SLURM_ARRAY_TASK_ID

case $seed in

    1)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=1/checkpoints/epoch=50-char_error_rate=0.1399-word_error_rate=0.1675.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=1/checkpoints/epoch=118-char_error_rate=0.1262-word_error_rate=0.1625.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/SAR_resnet31_nolrschedule_lr=1e-3_bsz=_seed=1/checkpoints/epoch=46-char_error_rate=0.1135-word_error_rate=0.1510.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/FPHTR_word_resnet31_lr=1e-4_bsz=32_seed=1/checkpoints/epoch=126-char_error_rate=0.0747-word_error_rate=0.1112.ckpt
        ;;
    2)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=2/checkpoints/epoch=101-char_error_rate=0.1372-word_error_rate=0.1673.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=2/checkpoints/epoch=120-char_error_rate=0.1205-word_error_rate=0.1534.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/SAR_resnet31_nolrschedule_lr=1e-3_bsz=_seed=2/checkpoints/epoch=40-char_error_rate=0.1155-word_error_rate=0.1474.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/FPHTR_word_resnet31_lr=1e-4_bsz=32_seed=2/checkpoints/epoch=69-char_error_rate=0.0742-word_error_rate=0.1145.ckpt

        ;;
    3)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=3/checkpoints/epoch=114-char_error_rate=0.1464-word_error_rate=0.1660.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=3/checkpoints/epoch=128-char_error_rate=0.1226-word_error_rate=0.1592.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/SAR_resnet31_nolrschedule_lr=1e-3_bsz=_seed=3/checkpoints/epoch=51-char_error_rate=0.1118-word_error_rate=0.1472.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/FPHTR_word_resnet31_lr=1e-4_bsz=32_seed=3/checkpoints/epoch=55-char_error_rate=0.0800-word_error_rate=0.1164.ckpt
        ;;
    4)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=4/checkpoints/epoch=79-char_error_rate=0.1219-word_error_rate=0.1551.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=4/checkpoints/epoch=76-char_error_rate=0.1304-word_error_rate=0.1640.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/SAR_resnet31_nolrschedule_lr=1e-3_bsz=_seed=4/checkpoints/epoch=36-char_error_rate=0.1185-word_error_rate=0.1502.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/FPHTR_word_resnet31_lr=1e-4_bsz=32_seed=4/checkpoints/epoch=38-char_error_rate=0.0848-word_error_rate=0.1203.ckpt
        ;;
    5)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=5/checkpoints/epoch=90-char_error_rate=0.1290-word_error_rate=0.1576.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=5/checkpoints/epoch=113-char_error_rate=0.1273-word_error_rate=0.1605.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/SAR_resnet31_nolrschedule_lr=1e-3_bsz=_seed=5/checkpoints/epoch=67-char_error_rate=0.1063-word_error_rate=0.1469.ckpt
        # TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/FPHTR_word_resnet31_lr=1e-4_bsz=32_seed=5/checkpoints/epoch=46-char_error_rate=0.0788-word_error_rate=0.1177.ckpt
        ;;
    *)
        echo "Invalid seed: $seed"
        exit 1
        ;;
esac

# LOG_DIR=/home/s4314719/master-thesis-exp/thesis/maml/lightning_logs/py38
LOG_DIR=/home/s4314719/master-thesis-exp/thesis/maml/lightning_logs
CACHE_DIR=/data/s4314719/thesis/metahtr/lightning_logs/cache_without_bad_segmentation

inner_lr_flag=""
if [ $use_msgd == "yes" ]; then
    main_model_arch="MetaHTR"
    model_desc="MAMLmSGD"
    inner_lr_flag="--initial_inner_lr $initial_inner_lr"
else
    main_model_arch="MAML"
    model_desc=$main_model_arch
    inner_lr_flag="--inner_lr $inner_lr"
fi

if [ $use_msgd == "yes" ]; then
    experiment_name="${model_desc}-${base_model_arch}${enc_version}_lr=${learning_rate}_initial-inner-lr=${initial_inner_lr}_nsteps=${num_inner_steps}_shots=${shots}_ways=${ways}_seed=${seed}"
else
    experiment_name="${model_desc}-${base_model_arch}${enc_version}_lr=${learning_rate}_inner-lr=${inner_lr}_nsteps=${num_inner_steps}_shots=${shots}_ways=${ways}_seed=${seed}"
fi

logdir="${LOG_DIR}/${experiment_name}"
logfile="${logdir}/train.out"
srcdir="${logdir}/src"

mkdir -p $logdir

# Copy job file
job_file=$(realpath $0)
cp $job_file $logdir

# Copy source code
mkdir -p $srcdir/metahtr
mkdir -p $srcdir/writer_code
cp *.py $srcdir
cp writer_code/*.py $srcdir/writer_code
cp metahtr/*.py $srcdir/metahtr

srun python -u main.py \
--main_model_arch $main_model_arch \
--base_model_arch $base_model_arch \
--trained_model_path $TRAINED_MODEL_PATH \
--log_dir $LOG_DIR \
--cache_dir $CACHE_DIR \
--experiment_name $experiment_name \
--learning_rate $learning_rate \
--inner_lr $inner_lr \
--weight_decay $weight_decay \
--num_nodes $SLURM_JOB_NUM_NODES \
--max_val_batch_size $max_val_batch_size \
--num_inner_steps $num_inner_steps \
--seed $seed \
--shots $shots \
--ways $ways \
--max_epochs $max_epochs \
--early_stopping_patience $early_stopping_patience \
--check_val_every_n_epoch $check_val_every_n_epoch \
--data_dir /data/s4314719/IAM \
--grad_clip 5 \
--num_workers 12 \
--use_image_augmentations \
--use_dropout \
$inner_lr_flag \
&> $logfile
# --track_grad_norm 2 \
# --use_cosine_lr_scheduler \
# --use_batch_stats_for_batchnorm \
# --precision 16 \
# --num_sanity_val_steps 1 \
#--freeze_batchnorm_gamma \
# --validate
# --save_all_checkpoints \
