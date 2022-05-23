#!/bin/bash

#SBATCH --job-name='code'
#SBATCH --partition=gpushort
#SBATCH --time=02:00:00
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
module load Python/3.9.5-GCCcore-10.3.0
source /data/s4314719/thesis-exp/env/bin/activate

seed=${SLURM_ARRAY_TASK_ID}

main_model_arch="WriterCodeAdaptiveModelNonEpisodic"
base_model_arch="fphtr"
# adaptation_method="cnn_output"
adaptation_method="conditional_batchnorm"
code_name="hinge"                  # size: 465
# code_name="quadhinge"              # size: 5,184
# code_name="cohinge"                # size: 10,000
# code_name="cochaincode-hinge"      # size: 64
# code_name="triplechaincode-hinge"  # size: 512
# code_name="delta-hinge"            # size: 780
learning_rate=1e-3
weight_decay=0
adaptation_num_hidden=128
batch_size=64
max_epochs=-1
early_stopping_patience=8


case $seed in

    1)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=1/checkpoints/epoch=50-char_error_rate=0.1399-word_error_rate=0.1675.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=1/checkpoints/epoch=118-char_error_rate=0.1262-word_error_rate=0.1625.ckpt
        ;;
    2)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=2/checkpoints/epoch=101-char_error_rate=0.1372-word_error_rate=0.1673.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=2/checkpoints/epoch=120-char_error_rate=0.1205-word_error_rate=0.1534.ckpt
        ;;
    3)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=3/checkpoints/epoch=114-char_error_rate=0.1464-word_error_rate=0.1660.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=3/checkpoints/epoch=128-char_error_rate=0.1226-word_error_rate=0.1592.ckpt
        ;;
    4)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=4/checkpoints/epoch=79-char_error_rate=0.1219-word_error_rate=0.1551.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=4/checkpoints/epoch=76-char_error_rate=0.1304-word_error_rate=0.1640.ckpt
        ;;
    5)
        # TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/SAR_resnet18_nolrschedule_lr=1e-3_bsz=_seed=5/checkpoints/epoch=90-char_error_rate=0.1290-word_error_rate=0.1576.ckpt
        TRAINED_MODEL_PATH=/home/s4314719/htr/lightning_logs/FPHTR_word_resnet18_lr=1e-4_bsz=32_seed=5/checkpoints/epoch=113-char_error_rate=0.1273-word_error_rate=0.1605.ckpt
        ;;
    *)
        echo "Invalid seed: $seed"
        exit 1
        ;;
esac

LOG_DIR=/home/s4314719/master-thesis-exp/thesis/writer_code/lightning_logs

log_adaptation_method=$(echo $adaptation_method | sed 's/_/-/g')
# experiment_name="${main_model_arch}-${base_model_arch}18_${code_name}_${log_adaptation_method}_lr=${learning_rate}_bsz=${batch_size}_wd=${weight_decay}_num-hidden=${adaptation_num_hidden}_seed=${seed}"
# experiment_name="${main_model_arch}-${base_model_arch}18_cnd-layernorm_${code_name}_${log_adaptation_method}_lr=${learning_rate}_bsz=${batch_size}_wd=${weight_decay}_num-hidden=${adaptation_num_hidden}_seed=${seed}"
experiment_name="${main_model_arch}-${base_model_arch}18_ada-bn_seed=${seed}"

logdir="$LOG_DIR/${experiment_name}"
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


srun python -u main_non-episodic.py \
--main_model_arch $main_model_arch \
--base_model_arch $base_model_arch \
--trained_model_path $TRAINED_MODEL_PATH \
--log_dir $LOG_DIR \
--experiment_name $experiment_name \
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--num_nodes $SLURM_JOB_NUM_NODES \
--seed $seed \
--code_name $code_name \
--adaptation_method $adaptation_method \
--batch_size $batch_size \
--adaptation_num_hidden $adaptation_num_hidden \
--max_epochs $max_epochs \
--early_stopping_patience $early_stopping_patience \
--data_dir /data/s4314719/IAM \
--num_workers 12 \
--use_image_augmentations \
&> $logfile
# --track_grad_norm 2 \
# --check_val_every_n_epoch 1 \
# --grad_clip 5 \
# --num_sanity_val_steps 1 \
#--use_cosine_lr_scheduler \
# --validate
# --save_all_checkpoints \
