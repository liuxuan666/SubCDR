import torch
import torch.utils.data as Data
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn import SGConv, GatedGraphConv
import numpy as np
from utils import *
from torch_dataset import *
from torch.nn.utils.rnn import pad_packed_sequence

torch.manual_seed(2022)
class SubEncoder(nn.Module):
    def __init__(self, in_drug, in_cline, out):
        super(SubEncoder, self).__init__()
        #---drug_layer
        self.dlayer = nn.GRU(in_drug, out, 1, batch_first = True)
        #self.batchd = nn.BatchNorm1d(19)
        #---cell line_layer
        self.clayer1 = nn.Conv1d(in_cline, in_cline, kernel_size=3, stride=1)
        self.clayer2 = nn.Conv1d(in_cline, in_cline, kernel_size=3, stride=2)
        self.batchc = nn.BatchNorm1d(8)
        #---map_coefficient
        self.weight = Parameter(torch.Tensor(out, out))
        self.act = nn.LeakyReLU(0.2)
        self.reset_para()
    
    def reset_para(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        glorot(self.weight)
        return
    
    def bilinearity(self, P, Q, act = True):
        score = torch.bmm(P, torch.matmul(self.weight, Q))
        return torch.sigmoid(score) if act else score
    
    def forward(self, x_drug, x_cline):
        #---drug_embeddings
        x_drug, _ = self.dlayer(x_drug)
        x_drug, _ = pad_packed_sequence(x_drug, batch_first=True) #, total_length=19)
        x_drug = self.act(x_drug)
        #x_drug = self.batchd(x_drug)
        #---cell line_embeddings
        x_cline = self.clayer1(x_cline)
        x_cline = self.act(x_cline)
        x_cline = self.clayer2(x_cline)
        x_cline = self.act(x_cline)
        x_cline = self.batchc(x_cline)    
        return self.bilinearity(x_drug, x_cline.permute(0,2,1))

class GloEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GloEncoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//2)
        self.fc2 = nn.Linear(in_channels//2, out_channels)
        self.batch = nn.BatchNorm1d(out_channels)
        self.act = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        x = self.batch(x)
        return x

class GraphEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GraphEncoder, self).__init__()
        self.conv = SGConv(in_channels, out_channels, K = 2)
        self.prelu = nn.PReLU(out_channels)
        self.batch = nn.BatchNorm1d(out_channels)
        self.pooling = GlobalPooling(["mean", "max"])
        
    def forward(self, x, edge_index, edge_weight, batch):
        x = self.conv(x, edge_index, edge_weight = edge_weight)
        x = self.batch(self.prelu(x))
        h = self.pooling(x, batch)
        return h

class Decoder(torch.nn.Module):
    def __init__(self, in_channels):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels//2)
        self.batch1 = nn.BatchNorm1d(in_channels//2)
        self.fc2 = nn.Linear(in_channels//2, in_channels//4)
        self.batch2 = nn.BatchNorm1d(in_channels//4)
        self.fc3 = nn.Linear(in_channels//4, 1)
        self.act = nn.ReLU()
        self.reset_parameters()
        
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
    def forward(self, h):
        h = self.act(self.fc1(h))
        h = self.batch1(h)
        #h = F.dropout(h, 0.1, training=self.training)
        h = self.act(self.fc2(h))
        h = self.batch2(h)
        #h = F.dropout(h, 0.2, training=self.training)
        h = self.fc3(h)
        return torch.sigmoid(h.squeeze(dim=-1))
    
class SubCDR(nn.Module):
    def __init__(self, SubEncoder, GraphEncoder, GloEncoder, Decoder):
        super(SubCDR, self).__init__()
        self.SubEncoder = SubEncoder
        self.GraphEncoder = GraphEncoder
        self.Decoder = Decoder
        self.GloEncoder = GloEncoder
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.SubEncoder)
        reset(self.GraphEncoder)
        reset(self.GloEncoder)
        reset(self.Decoder)
    
    def interaction_edge(self, adj, nan):
        a, b = torch.where(adj != nan)
        edge_index = torch.cat((a.unsqueeze(dim=-1), b.unsqueeze(dim=-1)), 1)
        edge_index = torch.cat((edge_index.T, edge_index[:, [1, 0]].T), 1)
        edge_weight = adj[a, b]
        edge_weight = torch.cat((edge_weight, edge_weight), 0)
        return edge_index, edge_weight
    
    def interaction_graph(self, maps):
        graphs = []
        node_num = maps[0].shape[0] + maps[0].shape[1]
        attributes = self.feature_initialize(node_num, 32)
        for item in maps:
        # 0.5 indicates that the interaction position is empty(0), because 0 is converted to 0.5 after sigmoid
            edge_index, edge_weight = self.interaction_edge(item, 0.5)
            graphs.append([attributes, edge_index, edge_weight])
        graph_batch = Data.DataLoader(dataset = GraphDataset(graphs_dict = graphs), \
                            collate_fn = collate, batch_size = len(graphs), shuffle = False)   
        return graph_batch
    
    def feature_initialize(self, node_num, max_dim):
        node_feat = torch.eye(node_num)
        node_feat = torch.cat((node_feat, torch.zeros([node_num, max_dim - node_num])), axis = 1) \
                    if (node_num < max_dim) else node_feat  
        return node_feat

    def forward(self, drug_sub, cline_sub, embed_glo):
        interaction_maps = self.SubEncoder(drug_sub, cline_sub)
        graphs = self.interaction_graph(interaction_maps)
        batch_graphs = graphs.__iter__().next()
        embed_sub = self.GraphEncoder(batch_graphs.x, batch_graphs.edge_index, \
                                       batch_graphs.edge_weight, batch_graphs.batch)
        embed_glo = self.GloEncoder(embed_glo)
        H = torch.cat((embed_sub, embed_glo), 1)
        pred = self.Decoder(H)
        return pred, interaction_maps    