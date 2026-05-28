import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
import sys
import os

# Define BBBP_Dataset structure based on what's expected
class BBBP_Dataset(Dataset):
    def __init__(self, csv_file, mapping_mode='helm_force'):
        self.df = pd.read_csv(csv_file)
        self.mapping_mode = mapping_mode
        # In the actual script, there's more initialization, but for size inspection,
        # we just need the length and potentially the split columns.

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return idx # Dummy

def make_loaders(dataset: BBBP_Dataset, batch_size: int, split_idx: int):
    # Logic from scripts/train_5split_ensemble_grid.py:
    # train_idx = dataset.df[dataset.df[f'split{split_idx}'] == 'train'].index
    # val_idx = dataset.df[dataset.df[f'split{split_idx}'] == 'val'].index
    # test_idx = dataset.df[dataset.df[f'split{split_idx}'] == 'test'].index
    
    train_idx = dataset.df[dataset.df[f'split{split_idx}'] == 'train'].index.tolist()
    val_idx = dataset.df[dataset.df[f'split{split_idx}'] == 'val'].index.tolist()
    test_idx = dataset.df[dataset.df[f'split{split_idx}'] == 'test'].index.tolist()
    
    train_loader = DataLoader(Subset(dataset, train_idx), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(Subset(dataset, val_idx), batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(Subset(dataset, test_idx), batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, len(train_idx), len(val_idx), len(test_idx)

csv_path = 'dataset/fasta_trainval_5splits.csv'
if not os.path.exists(csv_path):
    print(f"Error: {csv_path} not found")
    sys.exit(1)

ds = BBBP_Dataset(csv_path, mapping_mode='helm_force')
train_loader, val_loader, test_loader, train_count, val_count, test_count = make_loaders(ds, batch_size=32, split_idx=1)

print(f"Split 1 Sizes:")
print(f"Train: {train_count}")
print(f"Val: {val_count}")
print(f"Test: {test_count}")
print(f"Number of train batches (BS=32): {len(train_loader)}")
