import numpy as np
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem import BRICS
from rdkit import DataStructs
from rdkit.Chem import AllChem
import re

def BRICS_GetMolFrags(smi):
    mol = Chem.MolFromSmiles(smi)
    smarts = Chem.MolToSmiles(mol)
    mol = Chem.MolFromSmiles(smarts)
    #---mol Decompose    
    sub_smi = BRICS.BRICSDecompose(mol)
    sub_smi = [re.sub(r'\[\d+\*\]','*',item) for item in sub_smi]
    return sub_smi, smarts

# def BRICS_GetMolFrags(smi):
#     mol = Chem.MolFromSmiles(smi)
#     smarts = Chem.MolToSmiles(mol)
#     mol = Chem.MolFromSmiles(smarts)
#     #---mol Decompose    
#     mm = BRICS.BreakBRICSBonds(mol)
#     frags = Chem.GetMolFrags(mm, asMols = True)
#     sub_smi = [Chem.MolToSmiles(x, True) for x in frags]
#     sub_smi = [re.sub(r'\[\d+\*\]','*',item) for item in sub_smi]
#     return sub_smi, smarts

def FRL_GetMolFrags(smi):
    mol_t = Chem.MolFromSmiles(smi)
    smarts = Chem.MolToSmiles(mol_t)
    for i in mol_t.GetAtoms():
        i.SetIntProp("atom_idx", i.GetIdx())
    for i in mol_t.GetBonds():
        i.SetIntProp("bond_idx", i.GetIdx())
    ring_info = mol_t.GetRingInfo()
    bondrings = ring_info.BondRings() 
    if len(bondrings) == 0:
        bondring_list = []
    elif len(bondrings) == 1:  
        bondring_list = list(bondrings[0])
    else:
        bondring_list = list(bondrings[0]+bondrings[1])
    all_bonds_idx = [bond.GetIdx() for bond in mol_t.GetBonds()]
    none_ring_bonds_list = []
    for i in all_bonds_idx:
        if i not in bondring_list:
            none_ring_bonds_list.append(i)
    cut_bonds = []
    for bond_idx in none_ring_bonds_list:
        bgn_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetBeginAtomIdx()
        ebd_atom_idx = mol_t.GetBondWithIdx(bond_idx).GetEndAtomIdx()
        if mol_t.GetBondWithIdx(bond_idx).GetBondTypeAsDouble() == 1.0:
            if mol_t.GetAtomWithIdx(bgn_atom_idx).IsInRing()+mol_t.GetAtomWithIdx(ebd_atom_idx).IsInRing() == 1:
                t_bond = mol_t.GetBondWithIdx(bond_idx)
                t_bond_idx = t_bond.GetIntProp("bond_idx")
                cut_bonds.append(t_bond_idx)
    if len(cut_bonds) == 0 :
        return smi, smarts
    else:
        res = Chem.FragmentOnBonds(mol_t, cut_bonds)
        frags = Chem.GetMolFrags(res, asMols=True)
        sub_smi = [Chem.MolToSmiles(x, True) for x in frags]
    return sub_smi, smarts

class FP:
    """
    Molecular fingerprint class, useful to pack features in pandas df
    Parameters
    ----------
    fp : np.array
        Features stored in numpy array
    names : list, np.array
        Names of the features
    """

    def __init__(self, fp, names):
        self.fp = fp
        self.names = names

    def __str__(self):
        return "%d bit FP" % len(self.fp)

    def __len__(self):
        return len(self.fp)

def get_cfps(mol, radius=2, nBits=512, useFeatures=False, counts=False, dtype=np.float32):
    """Calculates circural (Morgan) fingerprint.
    Parameters
    ----------
    mol : rdkit.Chem.rdchem.Mol
    radius : float
        Fingerprint radius, default 2
    nBits : int
        Length of hashed fingerprint (without descriptors), default 1024
    useFeatures : bool
        To get feature fingerprints (FCFP) instead of normal ones (ECFP), defaults to False
    counts : bool
        If set to true it returns for each bit number of appearances of each substructure (counts). Defaults to false (fingerprint is binary)
    dtype : np.dtype
        Numpy data type for the array. Defaults to np.float32 because it is the default dtype for scikit-learn
    Returns
    -------
    Fingerprint (feature) object
    """
    arr = np.zeros((1,), dtype)

    if counts is True:
        info = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures,
                                                   bitInfo=info)
        DataStructs.ConvertToNumpyArray(fp, arr)
        arr = np.array([len(info[x]) if x in info else 0 for x in range(nBits)], dtype)
    else:
        DataStructs.ConvertToNumpyArray(
            AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nBits, useFeatures=useFeatures), arr)
    return FP(arr, range(nBits))

def get_Morgan(smiles):
    m = Chem.MolFromSmiles(smiles)
    Finger = get_cfps(m)
    fp = Finger.fp
    fp = fp.tolist()
    return fp 
