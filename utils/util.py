import pickle
import numpy as np
import os
import scipy.sparse as sp
import torch
from scipy.sparse import linalg
from sklearn import metrics
from scipy.stats import pearsonr
import random

class DataLoader(object):
    def __init__(self, xs, ys, batch_size, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.origin_xs,self.origin_ys = xs,ys
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys

    def set_xy(self,x,y):
        self.xs = x
        self.ys = y
        self.origin_xs, self.origin_ys = x, y

    def get_origin(self):
        return self.origin_xs,self.origin_ys

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                yield (x_i, y_i)
                self.current_ind += 1

        return _wrapper()

class StandardScaler():
    """
    Standard the input
    """

    def __init__(self,dimen_size, mean, std):
        self.mean = mean
        self.std = std
        self.dimen_size = dimen_size

    def transform(self, data):
        # 单维
        if self.dimen_size == 1:
            return (data - self.mean[0]) / self.std[0]
        # 多维
        resule_list = []
        for i in range(self.dimen_size):
            mean_reslut = (data[..., i] - self.mean[i]) / self.std[i]
            resule_list.append(np.expand_dims(mean_reslut, axis=-1))
        result = np.concatenate(resule_list, axis=-1)
        return result

    def inverse_transform(self, data):
        # 单维
        if self.dimen_size == 1:
            return (data * self.std[0]) + self.mean[0]
        # 多维
        std_tensor = torch.ones_like(data, requires_grad=False)
        mean_tensor = torch.zeros_like(data, requires_grad=False)
        for i in range(self.dimen_size):
            std_tensor[:, i, :, :] = self.std[i]
            mean_tensor[:, i, :, :] = self.mean[i]
        return (data * std_tensor) + mean_tensor



def sym_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()

def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat= sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()

def calculate_normalized_laplacian(adj):
    """
    # L = D^-1/2 (D-A) D^-1/2 = I - D^-1/2 A D^-1/2
    # D = diag(A 1)
    :param adj:
    :return:
    """
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian

def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

def load_adj(pkl_filename, adjtype):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    if adjtype == "scalap":
        adj = [calculate_scaled_laplacian(adj_mx)]
    elif adjtype == "normlap":
        adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
    elif adjtype == "symnadj":
        adj = [sym_adj(adj_mx)]
    elif adjtype == "transition":
        adj = [asym_adj(adj_mx)]
    elif adjtype == "doubletransition":
        adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx)),adj_mx]
    elif adjtype == "identity":
        adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
    else:
        error = 0
        assert error, "adj type not defined"
    return sensor_ids, sensor_id_to_ind, adj


def load_dataset(dataset_dir, batch_size, valid_batch_size= None, test_batch_size=None, transf=True):
    data = {}

    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']
        data['y_' + category] = cat_data['y']

    x_train = data['x_train']
    mean_list = []
    std_list = []
    dimen_size = x_train.shape[3] - 1 if len(x_train.shape) > 3 else 1
    for i in range(dimen_size):
        mean_list.append(x_train[..., i].mean())
        std_list.append(x_train[..., i].std())

    scaler = StandardScaler(dimen_size=dimen_size, mean=mean_list, std=std_list)

    # 标准化
    if transf:
        for category in ['train', 'val', 'test']:
            if dimen_size == 2:
                data['x_' + category] = scaler.transform(data['x_' + category])
            else:
                data['x_' + category][..., 0:-1] = scaler.transform(data['x_' + category][..., 0:-1])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size)
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size)
    data['scaler'] = scaler
    return data

def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds-labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels, null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mape(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)/labels
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def metric(pred, real):
    mae = masked_mae(pred,real,0.0).item()
    mape = masked_mape(pred,real,0.0).item()
    rmse = masked_rmse(pred,real,0.0).item()
    return mae,mape,rmse


def np_rmspe(y_true, y_pred):
    loss = np.sqrt(np.mean(np.square(((y_true - y_pred) / (y_true + np.mean(y_true)))), axis=0))
    return loss


# 计算mape，公式经过更改。每次除以标签值加上所有标签的均值，最后结果乘以2
def np_mape(y_true, y_pred):
    loss = np.abs(y_true - y_pred) / (y_true + np.mean(y_true))
    return np.mean(loss) * 2


def metrix_six(y_pred,y_test):
    mae_list = []
    mape_list = []
    rmse_list = []
    rmspe_list = []
    r2_list = []
    r_list = []

    for step in range(3):
        y_test_t = y_test[:,step]
        y_pred_t = y_pred[:,step]

        mae = metrics.mean_absolute_error(y_test_t, y_pred_t)
        # mape = metrics.mean_absolute_percentage_error(y_test_t, y_pred_t)
        mape = np_mape(y_test_t, y_pred_t)
        rmse = metrics.mean_squared_error(y_test_t, y_pred_t) ** 0.5
        rmspe = np_rmspe(y_test_t, y_pred_t)
        # rmspe2 = masked_rmspe(y_test_t,y_pred_t)
        r2 = metrics.r2_score(y_test_t, y_pred_t)
        r = pearsonr(y_test_t, y_pred_t)[0]


        # break
        # break

        mae_list.append(mae)
        mape_list.append(mape)
        rmse_list.append(rmse)
        rmspe_list.append(rmspe)
        r2_list.append(r2)
        r_list.append(r)

        # 计算平均值
    print(f'平均,{np.mean(mae_list):.3f},{np.mean(mape_list):.3f},'
          f'{np.mean(rmse_list):.3f},{np.mean(rmspe_list):.3f},'
          f'{np.mean(r2_list):.3f},{np.mean(r_list):.3f}')
    # mae = metrics.mean_absolute_error(y_test_t, y_pred_t)
    # mape = np_mape(y_test_t, y_pred_t)
    # rmse = metrics.mean_squared_error(y_test_t, y_pred_t) ** 0.5
    # rmspe = np_rmspe(y_test_t, y_pred_t)
    # r2 = metrics.r2_score(y_test_t, y_pred_t)
    # r = pearsonr(y_test_t, y_pred_t)[0]
    return np.mean(mae_list),np.mean(mape_list),np.mean(rmse_list),\
           np.mean(rmspe_list),np.mean(r2_list),np.mean(r_list),


def set_seed(seed):
    os.environ['PYTHONHASHSEED']=str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

