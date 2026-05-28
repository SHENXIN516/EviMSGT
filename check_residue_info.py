import pandas as pd
from rdkit import Chem

df = pd.read_csv('dataset/fasta_trainval_from_1_5splits.csv')
sequences = df['sequence'].head(3).tolist()

for i, seq in enumerate(sequences):
    mol = Chem.MolFromFASTA(seq)
    print(f"\nSequence {i+1}: {seq[:30]}...")
    if mol is None:
        print("  Failed to create molecule from FASTA")
        continue
    
    atoms = list(mol.GetAtoms())[:10]
    for j, atom in enumerate(atoms):
        res_info = atom.GetPDBResidueInfo()
        mon_info = atom.GetMonomerInfo()
        
        res_str = "None"
        if res_info:
            res_str = f"ResNum: {res_info.GetResidueNumber()}, ResName: {res_info.GetResidueName()}"
        
        mon_str = "None"
        if mon_info:
            mon_str = f"Type: {mon_info.GetMonomerType()}, Name: {mon_info.GetName()}"
            
        print(f"  Atom {j}: {atom.GetSymbol()} | PDBResInfo: {res_str} | MonomerInfo: {mon_str}")

# Check if residue IDs can be extracted
has_res_info = any(atom.GetPDBResidueInfo() is not None for atom in mol.GetAtoms())
print(f"\nResidue IDs extractable from metadata: {has_res_info}")
