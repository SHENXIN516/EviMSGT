#!/bin/bash
#SBATCH --job-name=evimsgt_repr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=48:00:00
#SBATCH --array=1-3%3
#SBATCH --output=slurm-representative-%A_%a.out
#SBATCH --error=slurm-representative-%A_%a.err

set -euo pipefail

TASKS=(6 9 12)

IDX=$((SLURM_ARRAY_TASK_ID - 1))
TASK_ID=${TASKS[$IDX]}

CONFIG_NAME=${CONFIG_NAME:-evi_combo_weight}
USE_EVIDENTIAL=${USE_EVIDENTIAL:-1}
SELECTION_METRIC=${SELECTION_METRIC:-combo}
CLASS_WEIGHT_MODE=${CLASS_WEIGHT_MODE:-balanced}
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
echo "USE_EVIDENTIAL=${USE_EVIDENTIAL}"
echo "SELECTION_METRIC=${SELECTION_METRIC}"
echo "CLASS_WEIGHT_MODE=${CLASS_WEIGHT_MODE}"
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
  --use_evidential "${USE_EVIDENTIAL}" \
  --selection_metric "${SELECTION_METRIC}" \
  --class_weight_mode "${CLASS_WEIGHT_MODE}" \
  --pos_mode "${POS_MODE}" \
  --out_dir "results/representative_search/${CONFIG_NAME}" \
  --ckpt_dir "ckpt/representative_search/${CONFIG_NAME}"
