#!/bin/bash

set -euo pipefail

seed=1
main_model_arch="WriterCodeAdaptiveModelNonEpisodic"
base_model_arch="fphtr"
learning_rate=1e-3
weight_decay=0
code_name="hinge"
adaptation_num_hidden=128
batch_size=128
max_epochs=1


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

LOG_DIR=/scratch/s4314719/master-thesis/thesis/writer_code/lightning_logs


python -u main_non-episodic.py \
--main_model_arch $main_model_arch \
--base_model_arch $base_model_arch \
--trained_model_path $TRAINED_MODEL_PATH \
--log_dir $LOG_DIR \
--learning_rate $learning_rate \
--weight_decay $weight_decay \
--code_name $code_name \
--seed $seed \
--batch_size $batch_size \
--adaptation_num_hidden $adaptation_num_hidden \
--max_epochs $max_epochs \
--data_dir /data/s4314719/IAM \
--num_workers 12 \
--track_grad_norm 2 \
--early_stopping_patience 10 \
--check_val_every_n_epoch 1 \
--num_sanity_val_steps 0 \
--use_image_augmentations \
# --grad_clip 5 \
# --num_sanity_val_steps 1 \
#--use_cosine_lr_scheduler \
# --validate
# --save_all_checkpoints \
