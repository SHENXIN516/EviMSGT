import torch
import os
import sys

# Setup environment to include scripts directory
sys.path.append('scripts')
from train_plat import BBBP_Dataset, build_model_config

def get_info(csv_path, split_col=None):
    try:
        # Removed 'root' argument as it seems not supported by this version of BBBP_Dataset
        ds = BBBP_Dataset(
            csv_path=csv_path,
            split_col=split_col,
            mapping_mode='helm_force',
            split_name='train' if split_col else None
        )
        if len(ds) == 0:
            return "Empty"
        data = ds[0]
        return {
            'x_dim': data.x.shape[1],
            'edge_attr_dim': data.edge_attr.shape[1] if data.edge_attr is not None else None,
            'pos_dim': data.pos.shape[1] if hasattr(data, 'pos') and data.pos is not None else None,
            'has_atom_z': hasattr(data, 'atom_z') and data.atom_z is not None,
            'num_classes': ds.num_classes,
            'label_type': data.y.dtype
        }
    except Exception as e:
        import traceback
        return f"{str(e)}\n{traceback.format_exc()}"

print("Training Dataset (split1):")
print(get_info('dataset/fasta_trainval_5splits.csv', 'split1'))

print("\nIndependent Dataset:")
print(get_info('dataset/fasta_independent_test.csv'))

# Model Gate Weight shape
ckpt_path = "ckpt/single_split/grid_split1_multiscale_evi1_kl1e-05_an5_valacc0.8063.pt"
if os.path.exists(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location='cpu')
    print("\nCKPT Gate shape:")
    weights = ckpt['model_state_dict'].get('cross_scale_gate.weight', 'Not found')
    print(weights.shape if not isinstance(weights, str) else weights)
    
    # Try building model from config
    try:
        from plat_model.model import MultiScaleGraphTransformer
        config = ckpt['model_config']
        model = MultiScaleGraphTransformer(**config)
        print("Model Gate shape (from config):")
        print(model.cross_scale_gate.weight.shape)
    except Exception as e:
        print(f"Error building model: {e}")
