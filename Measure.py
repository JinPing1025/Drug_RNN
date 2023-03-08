
from rdkit.Chem import MolFromSmiles, AllChem
from rdkit import DataStructs
from rdkit import Chem
from rdkit.Chem import Draw

def check_validity(smiles_list):
    total = len(smiles_list)
    valid_molecules =[]
    count = 0
    for sm in smiles_list:
        if MolFromSmiles(sm) != None and sm !='':
            valid_molecules.append(sm)
            count = count +1
    valid_score = count/total*100
    
    return valid_molecules, valid_score

def check_novelty(valid_molecules,data):
    novel_molecules=[]
    for i in valid_molecules: 
        if i not in data:
            novel_molecules.append(i)
    novel_score = (len(novel_molecules)/len(valid_molecules))*100
    return novel_score

def check_uniqueness(smiles_list):
    
    unique_molecules = list(set(smiles_list))
    unique_score = (len(unique_molecules)/len(smiles_list))*100
    
    return unique_molecules , unique_score

def draw_molecule(unique_molecules):
    mols=[]
    for smi in unique_molecules:
        mol = Chem.MolFromSmiles(smi)
        mols.append(mol)
    img = Draw.MolsToGridImage([m for m in mols if m is not None][:15], molsPerRow=5, subImgSize=(150, 150))
    return img

def smi_mol(gen_smi):
    mols = []
    for i in gen_smi:
        mol = Chem.MolFromSmiles(i)
        mols.append(mol)
    return mols





