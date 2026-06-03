#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
SLURM_SCRIPT="${PROJECT_ROOT}/scripts/run_representative_search.slurm"

CONFIG_NAMES=(
  evi_combo_noweight
  ce_combo_weight
  ce_combo_noweight
)
USE_EVIDENTIAL=(
  1
  0
  0
)
CLASS_WEIGHT_MODE=(
  none
  balanced
  none
)

SELECTION_METRIC="${SELECTION_METRIC:-combo}"
POLL_SECONDS="${POLL_SECONDS:-120}"

wait_for_job() {
  local job_id="$1"
  while squeue -h -j "${job_id}" | grep -q .; do
    echo "Job ${job_id} still running or pending: $(date)"
    squeue -j "${job_id}" -o "%.18i %.8T %.10M %.20R"
    sleep "${POLL_SECONDS}"
  done

  echo "Job ${job_id} left queue. Accounting summary:"
  sacct -j "${job_id}" --format=JobID,JobName%25,State,Elapsed,ExitCode --parsable2 || true
}

cd "${PROJECT_ROOT}"

for i in "${!CONFIG_NAMES[@]}"; do
  config="${CONFIG_NAMES[$i]}"
  evi="${USE_EVIDENTIAL[$i]}"
  class_weight="${CLASS_WEIGHT_MODE[$i]}"

  echo "Submitting ${config}: USE_EVIDENTIAL=${evi}, CLASS_WEIGHT_MODE=${class_weight}, SELECTION_METRIC=${SELECTION_METRIC}"
  submit_output="$(
    sbatch \
      --export=ALL,CONFIG_NAME="${config}",USE_EVIDENTIAL="${evi}",SELECTION_METRIC="${SELECTION_METRIC}",CLASS_WEIGHT_MODE="${class_weight}" \
      "${SLURM_SCRIPT}"
  )"
  echo "${submit_output}"
  job_id="$(echo "${submit_output}" | awk '{print $NF}')"
  wait_for_job "${job_id}"
done

echo "All representative configs finished."
python scripts/summarize_representative_search.py \
  --search_root results/representative_search \
  --output_dir results/representative_search_summary \
  --tasks 4,6,9,12
