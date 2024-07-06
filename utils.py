from math import sqrt
from scipy import stats
import torch
import matplotlib.pyplot as plt
import scipy.sparse as sp
import numpy as np

def rmse(y, f):
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse

def mse(y, f):
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def mse_cust(y, f):
    mse = ((y - f) ** 2)
    return mse

def pearson(y, f):
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def spearman(y, f):
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(y, f):
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z
    return ci

def draw_cust_mse(mse_dict):
    best_mse = []
    best_mse_title = []
    i = 0
    for (key, value) in mse_dict.items():
        if i < 10 or (i > 13 and i < 24):
            best_mse.append(value)
            best_mse_title.append(key)
        i += 1

    plt.bar(best_mse_title, best_mse)
    plt.xticks(rotation=90)
    plt.title('GE & METH')
    plt.ylabel('MSE')
    plt.savefig("Blind drug.png")

def draw_loss(train_losses, test_losses, title):
    plt.figure()
    plt.plot(train_losses, label='train loss')
    plt.plot(test_losses, label='test loss')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(title + ".png")  # should before show method

def draw_pearson(pearsons, title):
    plt.figure()
    plt.plot(pearsons, label='test pearson')

    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Pearson')
    plt.legend()
    # save image
    plt.savefig(title + ".png")  # should before show method

def normalize(mx):
    rowsum = np.array(mx.sum(1)) #会得到一个（2708,1）的矩阵
    r_inv = np.power(rowsum, -1).flatten() #得到（2708，）的元祖
    #在计算倒数的时候存在一个问题，如果原来的值为0，则其倒数为无穷大，因此需要对r_inv中无穷大的值进行修正，更改为0
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
                np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def get_edges(atom_message, bond_message, a2a, a2b):
    atom_first, atom_second, bond_first, bond_second = [], [], [], []
    for i in range(10):  # atom_message.size()[0]
        for j in range(a2a.size()[1]):
            if a2a[i][j] != 0:
                atom_first.append(i - 1)
                atom_second.append(a2a[i][j].item() - 1)
            if a2b[i][j] != 0:
                bond_first.append(i - 1)
                bond_second.append(a2b[i][j].item() - 1)
    atom_edges = torch.tensor([atom_first, atom_second])
    bond_edges = torch.tensor([bond_first, bond_second])

    atom_shape = (atom_message.size()[0], atom_message.size()[0])
    bond_shape = (bond_message.size()[0], bond_message.size()[0])

    atom_adj = sp.coo_matrix((np.ones(atom_edges.shape[0]), (atom_edges[:, 0], atom_edges[:, 1])),
                             shape=atom_shape, dtype=np.float32)
    bond_adj = sp.coo_matrix((np.ones(bond_edges.shape[0]), (bond_edges[:, 0], bond_edges[:, 1])),
                             shape=bond_shape, dtype=np.float32)

    atom_adj = atom_adj + atom_adj.T.multiply(atom_adj.T > atom_adj) - atom_adj.multiply(atom_adj.T > atom_adj)  # 构造无向图
    bond_adj = bond_adj + bond_adj.T.multiply(bond_adj.T > bond_adj) - bond_adj.multiply(bond_adj.T > bond_adj)  # 构造无向图

    atom_adj = normalize(atom_adj + sp.eye(atom_adj.shape[0]))  # 构造对角阵
    bond_adj = normalize(bond_adj + sp.eye(bond_adj.shape[0]))  # 构造对角阵

    atom_adj = sparse_mx_to_torch_sparse_tensor(atom_adj)
    bond_adj = sparse_mx_to_torch_sparse_tensor(bond_adj)
    return atom_adj, bond_adj