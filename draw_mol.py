# -*- coding: utf-8 -*-
from rdkit import Chem
from rdkit.Chem import AllChem,Draw
from rdkit.Chem.Draw import rdMolDraw2D
from IPython.display import SVG
from rdkit.Chem.Draw.MolDrawing import MolDrawing, DrawingOptions

mol = Chem.MolFromSmiles('CC(=O)Oc1ccccc1C(=O)O')
d2d = rdMolDraw2D.MolDraw2DSVG(200,100)
d2d.drawOptions().bondLineWidth = 2
d2d.DrawMolecule(mol)
d2d.FinishDrawing()
res = d2d.GetDrawingText()
with open('Aspirin.svg', 'w+') as result:
    result.write(res)
    
#%%
mol = Chem.MolFromSmiles('C[C@]1(CCCN1C2=NN3C=CC=C3C(=N2)NC4=NNC(=C4)C5CC5)C(=O)NC6=CN=C(C=C6)F')
patt = Chem.MolFromSmarts('*c1ccc(F)nc1')

hit_ats = mol.GetSubstructMatches(patt)
bond_lists = []
for i, hit_at in enumerate(hit_ats):
    hit_at = list(hit_at)
    bond_list = []
    for bond in patt.GetBonds():
        a1 = hit_at[bond.GetBeginAtomIdx()]
        a2 = hit_at[bond.GetEndAtomIdx()]
        bond_list.append(mol.GetBondBetweenAtoms(a1, a2).GetIdx())
    bond_lists.append(bond_list)
colours = [(0, 1, 0), (0, 1, 0), (0, 1, 0)]
atom_cols = {}
bond_cols = {}
atom_list = []
bond_list = []
for i, (hit_atom, hit_bond) in enumerate(zip(hit_ats, bond_lists)):
    hit_atom = list(hit_atom)
    for at in hit_atom:
        atom_cols[at] = colours[i%3]
        atom_list.append(at)
    for bd in hit_bond:
        bond_cols[bd] = colours[i%3]
        bond_list.append(bd)
d = rdMolDraw2D.MolDraw2DSVG(200, 100)
rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=atom_list,
                                       highlightAtomColors=atom_cols,
                                       highlightBonds=bond_list,
                                       highlightBondColors=bond_cols)
d.drawOptions().bondLineWidth = 1
d.FinishDrawing()
res = d.GetDrawingText()
with open('BMS-754807-highlight.svg', 'w+') as result:
    result.write(res)

# mol.HasSubstructMatch(patt)
# hit_at = mol.GetSubstructMatch(patt)
# hit_bond = []
# for bond in patt.GetBonds():   
#     aid1 = hit_at[bond.GetBeginAtomIdx()]
#     aid2 = hit_at[bond.GetEndAtomIdx()]
#     hit_bond.append(mol.GetBondBetweenAtoms(aid1, aid2).GetIdx())
# d2d = rdMolDraw2D.MolDraw2DSVG(200, 100)
# rdMolDraw2D.PrepareAndDrawMolecule(d2d, mol, highlightAtoms=list(hit_at), highlightBonds=hit_bond)
# d2d.drawOptions().bondLineWidth = 2
# d2d.FinishDrawing()
# res = d2d.GetDrawingText()
# with open('Ipatasertib.svg', 'w+') as result:
#     result.write(res)
#%%
from rdkit.Chem.Draw import SimilarityMaps
smiles1 = 'COC1=NC(=NC2=C1N=CN2[C@H]3[C@H]([C@@H]([C@H](O3)CO)O)O)N' #ZINC000000895218 (D-Aspartate)
smiles2 = 'C1=NC2=C(N=C(N=C2N1[C@H]3[C@H]([C@@H]([C@H](O3)CO)O)O)F)N' #ZINC000000895034 (L-Ser)
mol1 = Chem.MolFromSmiles(smiles1)
mol2 = Chem.MolFromSmiles(smiles2)
a = SimilarityMaps.GetSimilarityMapForFingerprint(mol2, mol1, SimilarityMaps.GetMorganFingerprint,size=(200,200))








