#!/bin/bash
#SBATCH --job-name=evimsgt_main
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=1-16%3
#SBATCH --output=slurm-main-config-%A_%a.out
#SBATCH --error=slurm-main-config-%A_%a.err

set -euo pipefail

# Main fixed configuration selected from the representative CE search.
# Tasks 6, 9, and 12 were already run in results/ce_hparam_search/lr3e-4_drop0p05_wd1e-5.
TASKS=(1 2 3 4 5 7 8 10 11 13 14 15 16 17 18 19)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
TASK_ID=${TASKS[$IDX]}

CONFIG_NAME="lr3e-4_drop0p05_wd1e-5"
LR_VALUE="3e-4"
DROPOUT_VALUE="0.05"
WEIGHT_DECAY_VALUE="1e-5"

EPOCHS=${EPOCHS:-150}
BATCH_SIZE=${BATCH_SIZE:-32}
SEEDS=${SEEDS:-10,20,30,40,50}
POS_MODE=${POS_MODE:-2d}

if [ -f /home/shenxin/miniconda3/etc/profile.d/conda.sh ]; then
  source /home/shenxin/miniconda3/etc/profile.d/conda.sh
elif [ -f /home/shenxin/.miniconda3/etc/profile.d/conda.sh ]; then
  source /home/shenxin/.miniconda3/etc/profile.d/conda.sh
else
  echo "Cannot find conda.sh under /home/shenxin/miniconda3 or /home/shenxin/.miniconda3" >&2
  exit 1
fi

conda activate bbbp-split
cd /home/shenxin/EviMSGT

echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "TASK_ID=${TASK_ID}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "LR=${LR_VALUE}"
echo "DROPOUT=${DROPOUT_VALUE}"
echo "WEIGHT_DECAY=${WEIGHT_DECAY_VALUE}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "SEEDS=${SEEDS}"
echo "POS_MODE=${POS_MODE}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/run_fasta_workflow_batch.py \
  --manifest_dataset_root /home/shenxin/benchmark/dataset \
  --x_min "${TASK_ID}" \
  --x_max "${TASK_ID}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR_VALUE}" \
  --weight_decay "${WEIGHT_DECAY_VALUE}" \
  --dropout "${DROPOUT_VALUE}" \
  --use_evidential 0 \
  --selection_metric combo \
  --class_weight_mode balanced \
  --pos_mode "${POS_MODE}" \
  --out_dir "results/ce_hparam_search/${CONFIG_NAME}" \
  --ckpt_dir "ckpt/ce_hparam_search/${CONFIG_NAME}"

