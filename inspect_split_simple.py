import pandas as pd
import math
import os
import sys

csv_path = 'dataset/fasta_trainval_5splits.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found")
    sys.exit(1)

df = pd.read_csv(csv_path)
split_idx = 1
col = f'split{split_idx}'

if col not in df.columns:
    print(f"Error: Column {col} not found in {csv_path}")
    print(f"Available columns: {df.columns.tolist()}")
    sys.exit(1)

train_count = len(df[df[col] == 'train'])
val_count = len(df[df[col] == 'val'])
test_count = len(df[df[col] == 'test'])

batch_size = 32
train_batches = math.ceil(train_count / batch_size)

print(f"Split 1 Sizes:")
print(f"Train: {train_count}")
print(f"Val: {val_count}")
print(f"Test: {test_count}")
print(f"Number of train batches (BS=32): {train_batches}")
