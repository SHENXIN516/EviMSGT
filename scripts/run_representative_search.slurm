#!/bin/bash
#SBATCH --job-name=evimsgt_repr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=1-16%3
#SBATCH --output=slurm-representative-%A_%a.out
#SBATCH --error=slurm-representative-%A_%a.err

set -euo pipefail

TASKS=(4 6 9 12)
CONFIG_NAMES=(
  evi_combo_weight
  evi_combo_noweight
  ce_combo_weight
  ce_combo_noweight
)
USE_EVIDENTIAL=(1 1 0 0)
CLASS_WEIGHT_MODE=(balanced none balanced none)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
TASK_INDEX=$((IDX % ${#TASKS[@]}))
CONFIG_INDEX=$((IDX / ${#TASKS[@]}))

TASK_ID=${TASKS[$TASK_INDEX]}
CONFIG_NAME=${CONFIG_NAMES[$CONFIG_INDEX]}
EVI=${USE_EVIDENTIAL[$CONFIG_INDEX]}
CW_MODE=${CLASS_WEIGHT_MODE[$CONFIG_INDEX]}

LR=${LR:-5e-4}
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
mkdir -p logs

echo "HOST=$(hostname)"
echo "DATE=$(date)"
echo "TASK_ID=${TASK_ID}"
echo "CONFIG_NAME=${CONFIG_NAME}"
echo "USE_EVIDENTIAL=${EVI}"
echo "CLASS_WEIGHT_MODE=${CW_MODE}"
echo "LR=${LR}"
echo "EPOCHS=${EPOCHS}"
echo "BATCH_SIZE=${BATCH_SIZE}"
echo "SEEDS=${SEEDS}"
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'available', torch.cuda.is_available())"

python scripts/run_fasta_workflow_batch.py \
  --manifest_dataset_root /home/shenxin/benchmark/dataset \
  --x_min "${TASK_ID}" \
  --x_max "${TASK_ID}" \
  --seeds "${SEEDS}" \
  --epochs "${EPOCHS}" \
  --batch_size "${BATCH_SIZE}" \
  --lr "${LR}" \
  --use_evidential "${EVI}" \
  --selection_metric combo \
  --class_weight_mode "${CW_MODE}" \
  --pos_mode "${POS_MODE}" \
  --out_dir "results/representative_search/${CONFIG_NAME}" \
  --ckpt_dir "ckpt/representative_search/${CONFIG_NAME}"
