from torch_geometric.data import InMemoryDataset, Batch
from torch_geometric import data as DATA
import torch.utils.data as data
import torch
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence,pack_sequence,pad_packed_sequence

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
#device = torch.device('cpu')

class PairsData(data.Dataset):
    def __init__(self, data1, data2):
        self.d1 = data1
        self.d2 = data2
        
    def __len__(self):
        return len(self.d1)
    
    def __getitem__(self, index):#返回的是tensor
        s1 = torch.FloatTensor(self.d1[index])
        s2 = torch.FloatTensor(self.d2[index])
        data = torch.cat((s1, s2), 0)
        return data.to(device)

class BatchData(data.Dataset):
    def __init__(self, data):
        self.data = data
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        d = torch.FloatTensor(self.data[idx])
        return d.to(device)

class GraphDataset(InMemoryDataset):
    def __init__(self, root='.', dataset='davis', transform=None, pre_transform=None, graphs_dict=None, dttype=None):
        super(GraphDataset, self).__init__(root, transform, pre_transform)
        self.dataset = dataset
        self.dttype = dttype
        self.process(graphs_dict)

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + f'_data_{self.dttype}.pt']

    def download(self):
        pass

    def _download(self):
        pass

    def _process(self):
        pass
#         if not os.path.exists(self.processed_dir):
#             os.makedirs(self.processed_dir)
    def process(self, graphs_dict):
        data_list = []
        for sample in graphs_dict:
            features, edge_index, weight = sample[0], sample[1], sample[2]
            # features = torch.Tensor(sample[0]).to(device)
            # edge_index = torch.LongTensor(sample[1])
            # weight = torch.Tensor(sample[2]).to(device)
            GraphData = DATA.Data(x=features, edge_index=edge_index, edge_weight=weight)
            data_list.append(GraphData)
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def collate(data_list):
    batchA = Batch.from_data_list([data for data in data_list])
    return batchA.to(device)


def collate_seq(data):
    seq_len = [s.size(0) for s in data] # length for seqs
    data = pad_sequence(data, batch_first=True)
    data = pack_padded_sequence(data, seq_len, batch_first=True, enforce_sorted=False)
    return data.to(device)