import torch
import numpy as np
import pandas as pd
from molFrags import *

def data_process():
    #--------data_load
    PATH = './data'
    Drug_file = '%s/drug_smiles.csv'%PATH
    Cell_line_file = '%s/cell line_GEP.csv'%PATH
    Gene_role_file = '%s/gene_role.csv'%PATH
    Drug_responses_file = '%s/GDSC2_fitted_dose_response_25Feb20.csv'%PATH
    IC50_threds_file = '%s/drug_threshold.txt'%PATH
    Mask_file = '%s/masked_cancer_gene/masked.csv'%PATH
    #--------data_preprocessing
    #---drugs preprocessing
    drug = pd.read_csv(Drug_file, sep = ',',header = 0, index_col = [0])
    #---get fragment features for all drug smiles
    drug_subfeat = {}; drug_fragments = {}; SMARTS = []
    max_len = 0
    for tup in zip(drug.index, drug['Isosmiles']):
        #---smiles to frags
        sub_smi, sm = BRICS_GetMolFrags(tup[1])
        max_len = len(sub_smi) if len(sub_smi) > max_len else max_len
        #---mols to fingerprints
        sub_features = [np.array(get_Morgan(item)) for item in sub_smi]
        drug_subfeat[str(tup[0])] = np.array(sub_features)
        SMARTS.append(sm)
        drug_fragments[str(tup[0])] = sub_smi
    drug_dim = drug_subfeat['Dasatinib'].shape[1]
    
    #---cell lines preprocessing
    gexpr_data = pd.read_csv(Cell_line_file, sep = ',', header = 0, index_col = [0])
    Mask = pd.read_csv(Mask_file, sep = ',', header = 0, index_col = [0])
    gexpr_data = gexpr_data * Mask
    gene_annotation = pd.read_csv(Gene_role_file, sep = ',', header = 0, index_col = [0])
    gene_types = list(set(gene_annotation['Role in Cancer']))
    cline_subfeat = {}
    type_count = gene_annotation['Role in Cancer'].value_counts() 
    cline_dim = max(type_count)
    #---get fragments for all cell line expressions
    for index, row in gexpr_data.iterrows():
        sub_gexpr = []
        for gt in gene_types:
            gt_gexpr = row[gene_annotation['Role in Cancer'] == gt]
            #---padding
            value = gt_gexpr.values
            padding = np.zeros((cline_dim - len(value)))
            sub_gexpr.append(list(np.concatenate((value, padding), axis = 0) \
                                 if len(value) < cline_dim else value)) 
        cline_subfeat[str(index)] = np.array(sub_gexpr)

    #---response scores
    task = 'regression' # task = 'classification'
    CDR = pd.read_csv(Drug_responses_file, sep = ',',header = 0, index_col = [0])
    CDR_pairs = []
    if task=='regression':
        for index, row in CDR.iterrows():
            if (row['COSMIC_ID'] in gexpr_data.index) and (row['DRUG_NAME'] in drug.index):
                CDR_pairs.append((row['COSMIC_ID'], row['DRUG_NAME'], row['LN_IC50']))
    else:
        drug2thred={}
        for line in open(IC50_threds_file).readlines()[1:]:
            drug2thred[str(line.split('\t')[2])]=float(line.strip().split('\t')[1])
        for index, row in CDR.iterrows():
            if row['DRUG_NAME'] in drug2thred.keys():
                binary_IC50 = 1 if row['LN_IC50'] < drug2thred[row['DRUG_NAME']] else -1
                CDR_pairs.append((row['COSMIC_ID'], drug.loc[row['DRUG_NAME'],'PubChem_ID'], binary_IC50))
                
    # ---Remove Duplicates
    CDR_pairs = pd.DataFrame(CDR_pairs)   
    CDR_pairs.sort_values(by=[0,1,2], inplace=True, ascending=[True, True, True])    
    CDR_pairs.drop_duplicates(subset=[0,1], inplace=True)
    print('Total %d CDR pairs across %d cell lines and %d drugs.'%(len(CDR_pairs), len(gexpr_data.index), len(drug.index)))
    CDR_pairs.columns = ['Cline', 'Drug', 'IC50']
    
    return drug_subfeat, cline_subfeat, CDR_pairs, drug_dim, drug_fragments, gene_types