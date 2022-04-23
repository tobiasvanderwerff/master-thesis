#!/bin/bash

set -euo pipefail


# TRAINED_MODEL_PATH=/data/s4314719/full_page_HTR/lightning_logs/FPHTR_word_resnet18_lr=3e-4_bsz=32_seed=3/checkpoints/epoch=64-char_error_rate=0.1500-word_error_rate=0.1738.ckpt
# TRAINED_MODEL_PATH=/data/s4314719/full_page_HTR/lightning_logs/FPHTR_word_resnet18_lr=3e-4_bsz=32_seed=1/checkpoints/epoch=98-char_error_rate=0.1346-word_error_rate=0.1693.ckpt
TRAINED_MODEL_PATH=/data/s4314719/htr/lightning_logs/FPHTR_word_resnet31_lr=2e-4_bsz=32_seed=1/checkpoints/epoch=39-char_error_rate=0.0865-word_error_rate=0.1227.ckpt

CACHE_DIR=/data/s4314719/thesis/writer_code/lightning_logs/cache_without_bad_segmentation

python main.py \
--base_model_arch "fphtr" \
--main_model_arch "WriterCodeAdaptiveModel" \
--data_dir /data/s4314719/IAM \
--trained_model_path $TRAINED_MODEL_PATH \
--cache_dir $CACHE_DIR \
--ways 4 \
--shots 8 \
--max_epochs 4 \
--learning_rate 1e-4 \
--num_sanity_val_steps 0 \
--check_val_every_n_epoch 2 \
--weight_decay 0 \
# --track_grad_norm 2 \
# --overfit_batches 1 \
# --grad_clip 1 \
