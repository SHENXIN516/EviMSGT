#!/bin/bash
set -euo pipefail
source /home/shenxin/miniconda3/etc/profile.d/conda.sh
conda activate bbbp-split
cd /home/shenxin/EviMSGT/scripts
LOG=/home/shenxin/EviMSGT/results/workflow_batch/run_batch_8_10_20260519_150245.log
: > "$LOG"
echo "[START] $(date)" | tee -a "$LOG"
python -u run_fasta_workflow_batch.py \
  --dataset_dir /home/shenxin/EviMSGT/dataset \
  --x_min 8 --x_max 10 \
  --seeds 10,20,30,40,50 \
  --train_ratio 0.8 --val_ratio 0.1 \
  --mapping_mode helm_force \
  --model multiscale \
  --use_evidential 1 \
  --batch_size 32 --epochs 200 --lr 5e-4 \
  --kl_weight 1e-5 --anneal_epochs 5 \
  --pos_mode 2d \
  --out_dir /home/shenxin/EviMSGT/results/workflow_batch \
  --ckpt_dir /home/shenxin/EviMSGT/ckpt/workflow_batch \
  |& tee -a "$LOG"
