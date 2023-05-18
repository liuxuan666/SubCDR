import math
import torch
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score
from scipy.stats import pearsonr
from typing import List, Optional, Union
from torch_geometric.nn import global_add_pool, global_mean_pool, global_max_pool
from torch import Tensor

def uniform(size, tensor):
    if tensor is not None:
        bound = 1.0 / math.sqrt(size)
        tensor.data.uniform_(-bound, bound)

def kaiming_uniform(tensor, fan, a):
    if tensor is not None:
        bound = math.sqrt(6 / ((1 + a ** 2) * fan))
        tensor.data.uniform_(-bound, bound)

def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)

def glorot_orthogonal(tensor, scale):
    if tensor is not None:
        torch.nn.init.orthogonal_(tensor.data)
        scale /= ((tensor.size(-2) + tensor.size(-1)) * tensor.var())
        tensor.data *= scale.sqrt()

def reset(nn):
    def _reset(item):
        if hasattr(item, 'reset_parameters'):
            item.reset_parameters()

    if nn is not None:
        if hasattr(nn, 'children') and len(list(nn.children())) > 0:
            for item in nn.children():
                _reset(item)
        else:
            _reset(nn)

def regression_metric(ytrue, ypred):
    rmse = mean_squared_error(y_true=ytrue, y_pred=ypred, squared=False)
    r2 = r2_score(y_true=ytrue, y_pred=ypred)
    r, p = pearsonr(ytrue, ypred)
    return rmse, r2, r

def classification_metric(yt, yp):
    precision, recall, _, = precision_recall_curve(yt, yp)
    aupr = -np.trapz(precision, recall)
    auc = roc_auc_score(yt, yp)
    #---f1,acc,recall, specificity, precision
    real_score=np.mat(yt)
    predict_score=np.mat(yp)
    sorted_predict_score = np.array(sorted(list(set(np.array(predict_score).flatten()))))
    sorted_predict_score_num = len(sorted_predict_score)
    thresholds = sorted_predict_score[np.int32(sorted_predict_score_num * np.arange(1, 1000) / 1000)]
    thresholds = np.mat(thresholds)
    thresholds_num = thresholds.shape[1]
    predict_score_matrix = np.tile(predict_score, (thresholds_num, 1))
    negative_index = np.where(predict_score_matrix < thresholds.T)
    positive_index = np.where(predict_score_matrix >= thresholds.T)
    predict_score_matrix[negative_index] = 0
    predict_score_matrix[positive_index] = 1
    TP = predict_score_matrix.dot(real_score.T)
    FP = predict_score_matrix.sum(axis=1) - TP
    FN = real_score.sum() - TP
    TN = len(real_score.T) - TP - FP - FN
    tpr = TP / (TP + FN)
    recall_list = tpr
    precision_list = TP / (TP + FP)
    f1_score_list = 2 * TP / (len(real_score.T) + TP - TN)
    accuracy_list = (TP + TN) / len(real_score.T)
    specificity_list = TN / (TN + FP)
    max_index = np.argmax(f1_score_list)
    f1_score = f1_score_list[max_index]
    accuracy = accuracy_list[max_index]
    specificity = specificity_list[max_index]
    recall = recall_list[max_index]
    precision = precision_list[max_index]
    return auc, aupr, # f1_score[0, 0], accuracy[0, 0], recall[0, 0], specificity[0, 0], precision[0, 0]

def set_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

class GlobalPooling(torch.nn.Module):
    r"""A global pooling module that wraps the usage of
    :meth:`~torch_geometric.nn.glob.global_add_pool`,
    :meth:`~torch_geometric.nn.glob.global_mean_pool` and
    :meth:`~torch_geometric.nn.glob.global_max_pool` into a single module.

    Args:
        aggr (string or List[str]): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            If given as a list, will make use of multiple aggregations in which
            different outputs will get concatenated in the last dimension.
    """
    def __init__(self, aggr: Union[str, List[str]]):
        super().__init__()

        self.aggrs = [aggr] if isinstance(aggr, str) else aggr

        assert len(self.aggrs) > 0
        assert len(set(self.aggrs) | {'sum', 'add', 'mean', 'max'}) == 4

    def forward(self, x: Tensor, batch: Optional[Tensor],
                size: Optional[int] = None) -> Tensor:
        """"""
        xs: List[Tensor] = []

        for aggr in self.aggrs:
            if aggr == 'sum' or aggr == 'add':
                xs.append(global_add_pool(x, batch, size))
            elif aggr == 'mean':
                xs.append(global_mean_pool(x, batch, size))
            elif aggr == 'max':
                xs.append(global_max_pool(x, batch, size))

        return xs[0] if len(xs) == 1 else torch.cat(xs, dim=-1)


    def __repr__(self) -> str:
        aggr = self.aggrs[0] if len(self.aggrs) == 1 else self.aggrs
        return f'{self.__class__.__name__}(aggr={aggr})'