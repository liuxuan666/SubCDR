import os
import numpy as np
import pandas as pd
import time
import torch
import torch.utils.data as Data
from utils import *
from MF import *
from models_classify import *
from torch_dataset import *
from sklearn.model_selection import KFold
import random
import argparse
from data_process import data_process

parser = argparse.ArgumentParser(description='Cancer_Drug_Response_Prediction_Independent')
parser.add_argument('--lr', dest = 'lr', type = float, default = 0.0001,)
parser.add_argument('--batch_sizes', dest = 'bs', type = int, default = 50)
parser.add_argument('--epoch', dest = 'ep', type = int, default = 100)
parser.add_argument('-output_dir', dest = 'o', default = "./output_dir/", help = "output directory")
args = parser.parse_args()

os.makedirs(args.o, exist_ok = True)
#---data process
start_time = time.time()
seed = 2022
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
drug_subfeat, cline_subfeat, CDR_pairs, drug_dim, drug_compo_elem, cline_compos_elem = data_process()

#%%---dataset_split and compile
train_size = 0.9
CV, Independent = np.split(CDR_pairs.sample(frac = 1, random_state = seed), [int(train_size * len(CDR_pairs))])

def getBinary(Tensors, thresh=-2):
    ones = torch.ones_like(Tensors)
    zeros = torch.zeros_like(Tensors)
    return torch.where(Tensors < thresh, ones, zeros)
    
#---data batchsize
def PairFeatures(pairs, drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem):
    drug_subs = []; cline_subs = [] 
    drug_glos = []; cline_glos = [] 
    drug_compos = []; cline_compos = []
    label = []
    for _, row in pairs.iterrows():   
        cline_subs.append(cline_subfeat[str(row[0])])
        drug_subs.append(drug_subfeat[str(row[1])])
        cline_glos.append(np.array(cline_glofeat.loc[row[0]]))
        drug_glos.append(np.array(drug_glofeat.loc[row[1]]))
        drug_compos.append([row[1], drug_compo_elem[str(row[1])]])
        cline_compos.append([row[0], cline_compos_elem])
        label.append(row[2])    
    return drug_subs, cline_subs, drug_glos, cline_glos, drug_compos, cline_compos, label

def BatchGenerate(pairs, drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, bs):    
    drug_subs, cline_subs, drug_glos, cline_glos, drug_compos, cline_compos, label\
               = PairFeatures(pairs, drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem)
    ds_loader = Data.DataLoader(BatchData(drug_subs), batch_size=bs, shuffle=False, collate_fn=collate_seq)
    cs_loader = Data.DataLoader(BatchData(cline_subs), batch_size=bs, shuffle=False)
    glo_loader = Data.DataLoader(PairsData(drug_glos, cline_glos), batch_size=bs, shuffle=False)
    label = torch.from_numpy(np.array(label, dtype='float32')).to(device)
    label = Data.DataLoader(dataset = Data.TensorDataset(label), batch_size=bs, shuffle=False)
    return ds_loader, cs_loader, glo_loader, drug_compos, cline_compos, label

def train(drug_loader_train, cline_loader_train, glo_loader_train, label_train):
    loss_train = 0
    Y_true, Y_pred=[], [] 
    for batch, (drug, cline, glo_feat, label) in enumerate(zip(drug_loader_train,\
                                          cline_loader_train, glo_loader_train, label_train)): 
        label = getBinary(label[0])
        pred, _ = model(drug.to(device), cline.to(device), glo_feat.to(device))
        optimizer.zero_grad()
        loss = myloss(pred, label)
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        Y_true += label.cpu().detach().numpy().tolist()
        Y_pred += pred.cpu().detach().numpy().tolist()
    auc, aupr = classification_metric(Y_true, Y_pred)
    print('train-loss=', loss_train/len(CV))
    print('train-AUC:' + str(round(auc, 4)) + ' train-AUPR:' + str(round(aupr, 4)))   

def test(drug_loader_test, cline_loader_test, glo_loader_test, label_test):
    loss_test = 0
    Y_true, Y_pred=[], []
    all_maps = []
    model.eval()
    with torch.no_grad():      
        for batch, (drug, cline, glo_feat, label) in enumerate(zip(drug_loader_test,\
                                                 cline_loader_test, glo_loader_test, label_test)): 
            label = getBinary(label[0])
            pred, maps = model(drug.to(device), cline.to(device), glo_feat.to(device)) 
            loss = myloss(pred, label)
            loss_test += loss.item()
            Y_true += label.cpu().detach().numpy().tolist()
            Y_pred += pred.cpu().detach().numpy().tolist()
    print('test-loss=', loss.item()/len(Independent))
    auc, aupr = classification_metric(Y_true, Y_pred)
    return auc, aupr, Y_true, Y_pred
        
#%%---traing and test
#---Building known matrix
CDR_known = CV.set_index(['Cline', 'Drug']).unstack('Cline')
CDR_known.columns = CDR_known.columns.droplevel()
#---MF    
CDR_matrix = np.array(CDR_known)
CDR_mask = 1-np.float32(np.isnan(CDR_matrix))
CDR_matrix[np.isnan(CDR_matrix)] = 0
drug_glofeat, cline_glofeat = svt_solve(A = CDR_matrix, mask = CDR_mask)
drug_glofeat = pd.DataFrame(drug_glofeat); cline_glofeat = pd.DataFrame(cline_glofeat)
drug_glofeat.index = list(CDR_known.index); cline_glofeat.index = list(CDR_known.columns)    
glo_dim = 2*drug_glofeat.shape[1]

Result = []
#Randomly shuffle samples
CV = CV.sample(frac = 1, random_state = seed)
Independent = Independent.sample(frac = 1, random_state = seed)
 
batch_sizes = args.bs
drug_loader_train, cline_loader_train, glo_loader_train, _, _, label_train = BatchGenerate(CV, \
                   drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, bs = batch_sizes)
drug_loader_test, cline_loader_test, glo_loader_test, dc_test, cc_test, label_test = BatchGenerate(Independent, \
                   drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, bs = batch_sizes)

 #%%   
model = SubCDR(SubEncoder(in_drug = drug_dim, in_cline = 8, out = 82), GraphEncoder(in_channels = 32, out_channels = 16), \
               GloEncoder(in_channels = glo_dim, out_channels = 128), Decoder(in_channels = 160)).to(device)   
optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)
myloss = torch.nn.BCELoss()

#---main
Result=[]
start = time.time() 
final_AUC = 0;final_AUPR = 0
for epoch in range(args.ep):
    print('epoch = %d'% (epoch+1))
    model.train()
    train(drug_loader_train, cline_loader_train, glo_loader_train, label_train)
    AUC, AUPR, Y_true, Y_pred = test(drug_loader_test, cline_loader_test, glo_loader_test, label_test)
    print('test auc: ' + str(round(AUC, 4)) + '  test aupr: ' + str(round(AUPR, 4)))
    if (AUC > final_AUC):
        final_AUC = AUC;final_AUPR = AUPR
        torch.save(model.state_dict(), args.o + "classification_model.pkl")
        
print('Final_AUC: ' + str(round(final_AUC, 4)) + '  Final_AUPR: ' + str(round(final_AUPR, 4)))
Result.append([final_AUC, final_AUPR])
#save_prediction_results  
odir =  args.o + "classification.txt"  
np.savetxt(odir, np.array(Result))
