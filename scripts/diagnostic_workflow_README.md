# EviMSGT diagnostic workflow

This folder contains three lightweight diagnostic scripts for the current benchmark stage.

## 1. Task performance and error analysis

```bash
python scripts/analyze_task_failures.py \
  --results_dir results/ce_hparam_search/lr3e-4_drop0p05_wd1e-5 \
  --dataset_root /home/shenxin/benchmark/dataset \
  --tasks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --error_split test \
  --output_dir results/task_failure_analysis
```

Outputs:

- `split_distribution.csv`: n/pos/neg/pos_rate/sequence length by task, seed, and split.
- `metric_summary.csv`: val/test/ind summary for finished tasks.
- `errors_test.csv`: FP/FN/correct samples from the val-selected checkpoint.

Use `--skip_errors` for a fast distribution-only pass.

## 2. Residue mapping audit

```bash
python scripts/audit_residue_mapping.py \
  --dataset_root /home/shenxin/benchmark/dataset \
  --results_dir results/ce_hparam_search/lr3e-4_drop0p05_wd1e-5 \
  --tasks 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19 \
  --seeds 10 \
  --pos_mode auto \
  --output_dir results/residue_mapping_audit
```

Outputs:

- `residue_mapping_audit_by_sample.csv`: sequence length, residue count, atom counts per residue, missing/merged flag.
- `residue_mapping_audit_summary.csv`: per-task mapping summary.

Use the same `--pos_mode` as the training run if you want to reuse existing graph caches.
When `--results_dir` contains finished workflow `metrics*.csv`, the split rows are audited from the
same `split_csv` and graph cache used during training. If no cached split CSV is found, the script
falls back to manifest rows and prints progress while rebuilding graphs.

## 3. Minimal explanation export

Use one val-selected checkpoint and its corresponding split CSV from `metrics*.csv`:

```bash
python scripts/export_multiscale_explanations.py \
  --ckpt_path /path/to/val_selected_checkpoint.pt \
  --csv_path /path/to/trainval_task_seed.csv \
  --split test \
  --max_samples 5 \
  --top_k 10 \
  --out_dir results/explanations/taskXX_seedYY
```

Outputs:

- `sample_predictions.csv`
- `top_atoms.csv`
- `top_residues.csv`
- `top_residue_edges.csv`
- `svg/*.svg` atom-highlight drawings
- `residue_edges/*.svg` residue-level connection drawings
