import numpy as np
import pandas as pd
import time
import torch
import torch.utils.data as Data
from utils import *
from MF import *
from models import *
from torch_dataset import *
from sklearn.model_selection import KFold
import random
import argparse
from data_process import data_process
import os

parser = argparse.ArgumentParser(description='Cancer_Drug_Response_Prediction_CV')
parser.add_argument('--lr', dest = 'lr', type = float, default = 0.0001,)
parser.add_argument('--batch_sizes', dest = 'bs', type = int, default = 50)
parser.add_argument('--epoch', dest = 'ep', type = int, default = 200)
parser.add_argument('-output_dir', dest = 'o', default = "./output_dir/", help = "output directory")
args = parser.parse_args()

os.makedirs(args.o, exist_ok = True)
dir_interation = args.o + "interaction_path_cv/"
os.makedirs(dir_interation, exist_ok = True)
#---data process
start_time = time.time()
seed = 2022
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
drug_subfeat, cline_subfeat, CDR_pairs, drug_dim, drug_compo_elem, cline_compos_elem = data_process()
if not os.path.isdir(args.o):
    os.mkdir(args.o)
    
#%%---dataset_split and compile
train_size = 0.9
CV, Independent = np.split(CDR_pairs.sample(frac = 1, random_state = seed), [int(train_size * len(CDR_pairs))])
kf = KFold(n_splits=5, shuffle=True, random_state=(seed))

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
        pred, _ = model(drug.to(device), cline.to(device), glo_feat.to(device))
        optimizer.zero_grad()
        loss = myloss(pred, label[0])
        loss.backward()
        optimizer.step()
        loss_train += loss.item()
        Y_true += label[0].cpu().detach().numpy().tolist()
        Y_pred += pred.cpu().detach().numpy().tolist()
    rmse, r2, pcc = regression_metric(Y_true, Y_pred)
    print('train-loss=', loss_train/len(train_set))
    print('train-RMSE:' + str(round(rmse, 4)) + ' train-R2:' + str(round(r2, 4)) +
        ' train-PCC:' + str(round(pcc, 4)))   
   
def validation(drug_loader_valid, cline_loader_valid, glo_loader_valid, label_valid):
    loss_validation = 0
    Y_true, Y_pred=[], []
    all_maps = []
    model.eval()
    with torch.no_grad():      
        for batch, (drug, cline, glo_feat, label) in enumerate(zip(drug_loader_valid,\
                                                 cline_loader_valid, glo_loader_valid, label_valid)):  
            pred, maps = model(drug.to(device), cline.to(device), glo_feat.to(device))
            loss = myloss(pred, label[0])
            loss_validation += loss.item()
            #map_store(all_maps, maps.cpu().detach().numpy().tolist())
            Y_true += label[0].cpu().detach().numpy().tolist()
            Y_pred += pred.cpu().detach().numpy().tolist()
    print('validation-loss=', loss.item()/len(validation_set))
    rmse, r2, pcc = regression_metric(Y_true, Y_pred)
    return rmse, r2, pcc, all_maps

def map_store(all_maps, maps):
    for item in maps:
        all_maps.append(item)
        
def save_maps(dc, cc, maps, fold):
    for (d, c, m) in zip(dc, cc, maps):
        fname = dir_interation + str(c[0]) + '&&' + str(d[0])+ '_' + str(fold) +'.csv'
        m = np.array(m)
        map_save = pd.DataFrame(m[:len(d[1]), :], index = d[1], columns = c[1])
        map_save.to_csv(fname)
        
#---Scenario: random cv
cv_data = CV
#%%---traing and valid
Result = []
fold = 1
for train_index, validation_index in kf.split(cv_data): 
    train_set, validation_set = cv_data.iloc[train_index,:], cv_data.iloc[validation_index,:]
    #---Building known matrix
    CDR_known = train_set.set_index(['Cline', 'Drug']).unstack('Cline')
    CDR_known.columns = CDR_known.columns.droplevel()
    #---MF    
    CDR_matrix = np.array(CDR_known)
    CDR_mask = 1-np.float32(np.isnan(CDR_matrix))
    CDR_matrix[np.isnan(CDR_matrix)] = 0
    drug_glofeat, cline_glofeat = svt_solve(A = CDR_matrix, mask = CDR_mask)
    drug_glofeat = pd.DataFrame(drug_glofeat); cline_glofeat = pd.DataFrame(cline_glofeat)
    drug_glofeat.index = list(CDR_known.index); cline_glofeat.index = list(CDR_known.columns)    
    glo_dim = 2*drug_glofeat.shape[1]
    
    #Randomly shuffle samples
    train_set = train_set.sample(frac = 1, random_state = seed)
    validation_set = validation_set.sample(frac = 1, random_state = seed)
     
    batch_sizes = args.bs
    drug_loader_train, cline_loader_train, glo_loader_train, _, _, label_train = BatchGenerate(train_set, \
                       drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, bs = batch_sizes)
    drug_loader_valid, cline_loader_valid, glo_loader_valid, dc_valid, cc_valid, label_valid = BatchGenerate(validation_set, \
                       drug_subfeat, cline_subfeat, drug_glofeat, cline_glofeat, drug_compo_elem, bs = batch_sizes)
    
    model = SubCDR(SubEncoder(in_drug = drug_dim, in_cline = 8, out = 82), GraphEncoder(in_channels = 32, out_channels = 16), \
                   GloEncoder(in_channels = glo_dim, out_channels = 128), Decoder(in_channels = 160)).to(device)   
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr, weight_decay = 1e-4)
    myloss = torch.nn.HuberLoss()

    #---main
    start = time.time() 
    Final_RMSE = 99; Final_R2 = 0; Final_PCC = 0
    for epoch in range(args.ep):
        print('epoch = %d'% (epoch+1))
        model.train()
        train(drug_loader_train, cline_loader_train, glo_loader_train, label_train)
        rmse, r2, pcc, all_maps = validation(drug_loader_valid, cline_loader_valid, glo_loader_valid, label_valid)
        print('valid-RMSE:' + str(round(rmse, 4)) + ' valid-R2:' + str(round(r2, 4)) +
             ' valid-PCC:' + str(round(pcc, 4)))  
        if(rmse < Final_RMSE):
            Final_RMSE = rmse; Final_R2 = r2; Final_PCC = pcc
            #save_maps(dc_valid, cc_valid, all_maps, fold)
            torch.save(model.state_dict(), args.o + "cv_"+ str(cv_scenario) + "_model.pkl")    
    print('\nFold = %s, Final_RMSE=%s, Final_R2=%s, Final_PCC=%s \n'%(fold, Final_RMSE, Final_R2, Final_PCC))
    Result.append([Final_RMSE, Final_R2, Final_PCC])
    fold = fold + 1
#%%
#save_prediction_results  
odir =  args.o + "cv_"+ str(cv_scenario) + "_results.txt"  
np.savetxt(odir, np.array(Result))
