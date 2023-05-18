# -*- coding: utf-8 -*-
import pandas as pd
import os

Classification_file = 'Classification.csv'
Cline_to_TumourType_file = 'Cline_to_TumourType.csv'
Gene_to_TumourType_file = 'Gene_to_TumourType.csv'
expmap_file = '../cell line_GEP.csv'

Classif = pd.read_csv(Classification_file, sep = ',', header = 0)
Cline_Tumour = pd.read_csv(Cline_to_TumourType_file, sep = ',', header = 0)
Gene_Tumour = pd.read_csv(Gene_to_TumourType_file, sep = ',', header = 0)
expmap = pd.read_csv(expmap_file, sep = ',', header = 0, index_col=0)
expmap.loc[:,:] = 0
gene_cline = {}
all_cline = Cline_Tumour['COSMIC_ID'].tolist()
for index, row in Gene_Tumour.iterrows():
    if pd.isnull(row[2]):
        gene_cline[row[0]] = set(all_cline)
    else:
        tumours = str(row[2]).split(";")
        types = Classif[Classif["TumourType"].isin(tumours)]
        filters = types['classification'].tolist()
        res = []
        for i in filters:
            if i in Cline_Tumour["Cancer type"].tolist():
                clines = Cline_Tumour[Cline_Tumour["Cancer type"]==i]
            elif i in Cline_Tumour["Tissue"].tolist():
                clines = Cline_Tumour[Cline_Tumour["Tissue"]==i]
            assert clines.shape[0] != 0 
            res.extend(clines['COSMIC_ID'].tolist())
        gene_cline[row[0]] = set(res)

for j in gene_cline.keys():
    for i in gene_cline[j]: 
        if i in list(expmap.index):
            expmap.loc[i,j] = 1

expmap.to_csv('masked.csv')
