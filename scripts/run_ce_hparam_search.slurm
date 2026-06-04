#!/bin/bash
#SBATCH --job-name=evimsgt_ce_hp
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=1-48%3
#SBATCH --output=slurm-ce-hparam-%A_%a.out
#SBATCH --error=slurm-ce-hparam-%A_%a.err

set -euo pipefail

TASKS=(6 9 12)
LRS=(1e-4 3e-4 5e-4 1e-3)
DROPOUTS=(0.05 0.1 0.2 0.1)
WEIGHT_DECAYS=(1e-5 1e-5 1e-4 0)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
TASK_INDEX=$((IDX % ${#TASKS[@]}))
CONFIG_INDEX=$((IDX / ${#TASKS[@]}))
LR_INDEX=$((CONFIG_INDEX / ${#DROPOUTS[@]}))
REG_INDEX=$((CONFIG_INDEX % ${#DROPOUTS[@]}))

TASK_ID=${TASKS[$TASK_INDEX]}
LR_VALUE=${LRS[$LR_INDEX]}
DROPOUT_VALUE=${DROPOUTS[$REG_INDEX]}
WEIGHT_DECAY_VALUE=${WEIGHT_DECAYS[$REG_INDEX]}

LR_TAG=${LR_VALUE//./p}
DROPOUT_TAG=${DROPOUT_VALUE//./p}
WEIGHT_DECAY_TAG=${WEIGHT_DECAY_VALUE//./p}
CONFIG_NAME="lr${LR_TAG}_drop${DROPOUT_TAG}_wd${WEIGHT_DECAY_TAG}"

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
