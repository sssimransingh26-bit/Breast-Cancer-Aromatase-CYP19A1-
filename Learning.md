# Learning Notes

## Aromatase
-Aromatase (CYP19A1) is an enzyme that converts androgens into estrogens. It's a drug target for breast cancer
-Many breast cancers are "estrogen-receptor positive" — meaning estrogen literally feeds the tumor's growth. So if you can block aromatase, you reduce estrogen production, and the tumor starves.


## What are Fingerprints?
- A way to convert molecules into numbers for ML
- Binary vector — 1 if substructure present, 0 if not

## Fingerprint Types
- MACCS (166-bit) — fixed checklist of substructures
- Morgan — circular, looks at atom neighborhoods
....

## Molecules and SMILES

### What is SMILES?
SMILES (Simplified Molecular Input Line Entry System) is a way to 
represent a molecule as a text string.

Examples:
- Water → O
- Ethanol → CCO
- Aspirin → CC(=O)Oc1ccccc1C(=O)O

Each letter is an atom
Letters next to each other means they are bonded.
— RDKit reads SMILES syntax for us.


### Loading a molecule in RDKit
from rdkit import Chem
mol = Chem.MolFromSmiles('CCO')  

### Basic properties
mol.GetNumAtoms()  # number of atoms
mol.GetNumBonds()  # number of bonds

### Visualizing
from rdkit.Chem import Draw
Draw.MolToImage(mol)

the above line returns image of molecule

## Morgan Fingerprints

### What is a Morgan Fingerprint?
A Morgan fingerprint is a way to convert a chemical molecule into a numerical vector that machine learning models can understand.
For each atom it looks at its neighborhood within a certain radius.

- Radius 0 — just the atom itself
- Radius 1 — atom + immediate neighbors  
- Radius 2 — atom + neighbors + neighbors of neighbors

Think of a Morgan fingerprint as a unique barcode for a molecule.
It captures important structural patterns such as rings, chains, and functional groups and converts them into a fixed-length vector of 0s and 1s.

### Code
from rdkit.Chem import AllChem
import numpy as np

fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
arr = np.array(fp)

### Key insight
More complex molecule = more bits set to 1
- Ethanol (CCO) → 6 bits set
- Aspirin → 24 bits set

##  MACCS Keys and Feature Matrix

### MACCS Keys
MACCS Keys (Molecular ACCess System Keys) are another way to convert a molecule into numbers for machine learning.
Fixed checklist of 167 predefined substructures.
Every molecule gets checked against the same 167 questions — 1 if present, 0 if not.

**Morgan Fingerprints**
Instead of predefined questions, the algorithm automatically looks at the person's features and creates its own description.

from rdkit.Chem import MACCSkeys
fp_maccs = MACCSkeys.GenMACCSKeys(mol)
arr_maccs = np.array(fp_maccs)

### MACCS vs Morgan
- MACCS — 167 bits, fixed questions, more interpretable
- Morgan — 2048 bits, based on molecule structure, captures more information

## Fetching Real Data from ChEMBL

### What is ChEMBL?
Public database of bioactivity data from lab experiments.
Target ID for human aromatase: CHEMBL1978

### Fetching data
from chembl_webresource_client.new_client import new_client

activity = new_client.activity
results = activity.filter(
    target_chembl_id='CHEMBL1978',
    standard_type='IC50'
)
df = pd.DataFrame.from_records(results)

### Columns we care about
- canonical_smiles — molecule structure
- standard_value — raw IC50 value
- pchembl_value — target variable (-log10 of IC50, higher = more potent)

### Cleaning
- Started with 4414 rows
- Dropped 565 rows with missing pchembl_value
- Final dataset: 3849 molecules
- pchembl_value range: 4 to 11, mean 6.28




