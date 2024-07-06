# import random
#
# import networkx as nx
# import numpy as np
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
# from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
#
# from featurization import mol2graph, index_select_ND, BatchGRU
# from smile2Vec import smile2Vec
# from utils import get_edges
#
#
# class GAT_GCN_Transformer_meth_ge_mut(torch.nn.Module):
#     def __init__(self, n_output=1, num_features_xd=119, num_features_xs=132,
#                  n_filters=32, layer_smile=3, hidden_dim=1500, output_dim=128, dropout=0.2):
#
#         super(GAT_GCN_Transformer_meth_ge_mut, self).__init__()
#
#         # atom
#         self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=num_features_xd, nhead=1, dropout=0.5)
#         self.ugformer_layer_1 = nn.TransformerEncoder(self.encoder_layer_1, 1)
#         self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
#         self.relu = nn.ReLU()
#         self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=num_features_xd * 10, nhead=1, dropout=0.5)
#         self.ugformer_layer_2 = nn.TransformerEncoder(self.encoder_layer_2, 1)
#         self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
#         self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, hidden_dim)
#         self.fc_g2 = torch.nn.Linear(hidden_dim, output_dim)
#         self.dropout = nn.Dropout(dropout)
#
#         # bond
#         self.encoder_layer1_1 = nn.TransformerEncoderLayer(d_model=num_features_xs, nhead=1, dropout=0.5)
#         self.ugformer_layer1_1 = nn.TransformerEncoder(self.encoder_layer1_1, 1)
#         self.fc_bond = nn.Linear(num_features_xs, num_features_xs * 10)
#         self.relu = nn.ReLU()
#         self.encoder_layer2_2 = nn.TransformerEncoderLayer(d_model=num_features_xs * 10, nhead=1, dropout=0.5)
#         self.ugformer_layer2_2 = nn.TransformerEncoder(self.encoder_layer2_2, 1)
#         self.fc_g11 = torch.nn.Linear(num_features_xs * 2 * 10, hidden_dim)
#         self.fc_g22 = torch.nn.Linear(hidden_dim, output_dim)
#
#         # cell line mut feature
#         self.conv_xt_mut_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
#         self.pool_xt_mut_1 = nn.MaxPool1d(3)
#         self.conv_xt_mut_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
#         self.pool_xt_mut_2 = nn.MaxPool1d(3)
#         self.conv_xt_mut_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
#         self.pool_xt_mut_3 = nn.MaxPool1d(3)
#         self.fc1_xt_mut = nn.Linear(2944, output_dim)
#
#         # cell line meth feature
#         self.conv_xt_meth_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
#         self.pool_xt_meth_1 = nn.MaxPool1d(3)
#         self.conv_xt_meth_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
#         self.pool_xt_meth_2 = nn.MaxPool1d(3)
#         self.conv_xt_meth_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
#         self.pool_xt_meth_3 = nn.MaxPool1d(3)
#         self.fc1_xt_meth = nn.Linear(1280, output_dim)
#
#         # cell line ge feature
#         self.conv_xt_ge_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
#         self.pool_xt_ge_1 = nn.MaxPool1d(3)
#         self.conv_xt_ge_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
#         self.pool_xt_ge_2 = nn.MaxPool1d(3)
#         self.conv_xt_ge_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
#         self.pool_xt_ge_3 = nn.MaxPool1d(3)
#         self.fc1_xt_ge = nn.Linear(4224, output_dim)
#
#         # smile2Vec GRU
#         self.smile2Vec = smile2Vec(np.load("data/smi_embedding_matrix.npy"))
#         self.layer_smile = layer_smile
#         self.fc_smile = nn.Linear(150, output_dim)
#         self.smile_W = nn.ModuleList([nn.Linear(output_dim, output_dim)
#                                       for _ in range(layer_smile)])
#
#         # CMPNN
#         # Input
#         atom_fdim = num_features_xd  # 133
#         self.hidden_size = 300
#         self.bias = False
#         self.dropout_layer = nn.Dropout(p=0.1)
#         self.W_i_atom = nn.Linear(atom_fdim, self.hidden_size, bias=self.bias)
#         bond_fdim = num_features_xs  # 147
#         self.W_i_bond = nn.Linear(bond_fdim, self.hidden_size, bias=self.bias)
#         self.act_func = nn.ReLU()
#         w_h_input_size_atom = self.hidden_size + bond_fdim
#
#         self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)
#
#         w_h_input_size_bond = self.hidden_size
#
#         self.depth = 2
#         for depth in range(self.depth):
#             self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)
#
#         self.W_o = nn.Linear(self.hidden_size * 4, self.hidden_size*2)
#         # not share parameter between node and edge so far
#         self.W_nout = nn.Linear(atom_fdim + self.hidden_size, self.hidden_size)
#         self.W_eout = nn.Linear(atom_fdim + self.hidden_size, self.hidden_size)
#
#         self.gru = BatchGRU(self.hidden_size*2)
#
#         self.lr = nn.Linear(self.hidden_size * 4 + atom_fdim, self.hidden_size*2, bias=self.bias)
#
#         # combined layers
#         self.fc1 = nn.Linear(4 * output_dim + 600, 1024)
#         self.batchnorm1 = nn.BatchNorm1d(1024)
#         self.relu1 = nn.LeakyReLU()
#         self.fc2 = nn.Linear(1024, 256)
#         self.out = nn.Linear(256, n_output)
#
#     def forward(self, atom, bond, edge_index, a2a, a2b, a_scope, b_scope, b2a, b2revb, \
#                 batch_atom, batch_bond, smiles_, smiles, target_mut, target_meth, target_ge):
#
#         # smiles_, smiles = data.smiles_seq, data.smiles
#         # atom, bond, edge_index, a2a, a2b, a_scope, b_scope, b2a, b2revb, batch_atom, batch_bond \
#         #     = graph_process(data.x, data.xc, data.edge_index2, data.c_size, data.b_size, data.edge_index1, data.batch)
#
#         # mol_graph = mol2graph(smiles)
#         # f_atoms, f_bonds, a2b, b2a, b2revb, a_scope, b_scope, bonds = mol_graph.get_components()
#         # f_atoms, f_bonds, a2b, b2a, b2revb = (
#         #     f_atoms.cuda(), f_bonds.cuda(),
#         #     a2b.cuda(), b2a.cuda(), b2revb.cuda())
#
#         # Transformer_atom
#         atom = torch.unsqueeze(atom, 1)  # torch.Size([4173, 119])-->torch.Size([4173, 1, 119])
#         atom = self.ugformer_layer_1(atom)  # torch.Size([4173, 1, 119])
#         atom = torch.squeeze(atom, 1)  # torch.Size([4173, 119])
#
#         # Transformer_bond
#         # bond = torch.unsqueeze(bond, 1)  # torch.Size([8655, 1, 132])
#         # bond = self.ugformer_layer1_1(bond)  # torch.Size([8655, 1, 132])
#         # bond = torch.squeeze(bond, 1)  # torch.Size([8655, 132])
#
#         # get_edges
#         # atom_adj, bond_adj = get_edges(atom, bond, a2a, a2b)
#
#         # Input 首先输入网络后先进行了线性层
#         input_atom = self.W_i_atom(atom)  # torch.Size([4173, 300])
#         input_atom = self.act_func(input_atom)  # torch.Size([4173, 300])
#         message_atom = input_atom.clone()  # torch.Size([4173, 300])
#
#         input_bond = self.W_i_bond(bond)  # torch.Size([9129, 132]) -->torch.Size([9129, 300])
#         input_bond = self.act_func(input_bond)  # torch.Size([9129, 300])
#         message_bond = input_bond.clone()  # torch.Size([9129, 300])
#
#         # message_bond = torch.cat((message_bond, input_bond), dim=1)
#
#         # 使用消息聚合模块聚集临近节点的信息
#         nei_message = self.select_neighbor_and_aggregate(message_atom, a2a)  # 根据边聚合  torch.Size([4173, 300])  torch.Size([4173, 4])--?torch.Size([4173, 300])
#         nei_attached_message = self.select_neighbor_and_aggregate(input_bond, a2b)  # 根据边聚合  torch.Size([9129, 300])  torch.Size([4173, 4])--?torch.Size([4173, 300])
#         # atom_message = torch.cat((nei_message, nei_attached_message), dim=1)  # torch.Size([4173, 600])
#         atom_message = nei_message + nei_attached_message
#         # Message passing
#         for depth in range(self.depth):  # 首先原子的特征维度为[1566, 133]   边的特征维度为 [3347, 147]， 首先得到每个原子上面的边以及维度，然后将得到的新特征加到原来的原子特征上
#             agg_message = index_select_ND(message_bond, a2b)  # torch.Size([4173, 4, 300])
#             agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]  # torch.Size([4173, 300])*torch.Size([4173, 300])->torch.Size([4173, 300])
#             message_atom = atom_message + agg_message  # torch.Size([4173, 300])
#
#             # directed graph
#             rev_message = input_bond[b2revb]  # torch.Size([9129, 300])
#             # rev_message = torch.cat((rev_message, input_bond[b2a[b2revb]]), dim=1)
#             rev_message = rev_message + input_bond[b2a[b2revb]]
#             message_bond = message_atom[b2a] - rev_message  # torch.Size([9129, 300])
#
#             message_bond = self._modules[f'W_h_{depth}'](message_bond)  # liner torch.Size([9129, 300])
#             message_bond = self.dropout_layer(self.act_func(message_bond))
#
#         agg_message = index_select_ND(message_bond, a2b)  # torch.Size([4173, 4, 300])
#         agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]  # torch.Size([4173, 300])
#         message_atom = message_atom + agg_message
#
#         # 交叉传递
#         message_atom = self.aggreate_to_atom_fea(message_atom, a2a, atom,
#                                                  linear_module=self.W_nout)  # torch.Size([4173, 300])
#         message_bond = self.aggreate_to_atom_fea(message_bond, a2b, atom,
#                                                  linear_module=self.W_eout)  # torch.Size([4173, 300])
#
#         agg_message = self.lr(torch.cat([agg_message, message_atom, message_bond, atom_message, atom], 1))  # torch.Size([4173, 300])
#
#         agg_message = self.gru(agg_message, a_scope)  # torch.Size([4173, 600])
#
#         atom_hiddens = self.dropout_layer(self.act_func(self.W_o(agg_message)))  # torch.Size([4173, 300])
#
#         # Readout
#         mol_vecs = []
#         for i, (a_start, a_size) in enumerate(a_scope):
#             cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
#             mol_vecs.append(cur_hiddens.mean(0))
#         mol_vecs = torch.stack(mol_vecs, dim=0)  # torch.Size([128, 300])
#
#         # # atom
#         # atom = torch.unsqueeze(atom, 1)
#         # atom = self.ugformer_layer_1(atom)
#         # atom = torch.squeeze(atom, 1)
#         # # residual = self.linear(atom)
#         # atom = self.conv1(atom, edge_index)
#         # atom = self.relu(atom)
#         # # atom = atom + residual
#         # atom = torch.unsqueeze(atom, 1)
#         # atom = self.ugformer_layer_2(atom)
#         # atom = torch.squeeze(atom, 1)
#         # # residual = self.linear2(atom)
#         # atom = self.conv2(atom, edge_index)
#         # atom = self.relu(atom)
#         # # atom = atom + residual
#         # atom = torch.cat([gmp(atom, batch_atom), gap(atom, batch_atom)],
#         #                  dim=1)  # apply global max pooling (gmp) and global mean pooling (gap)
#         # atom = self.relu(self.fc_g1(atom))
#         # atom = self.dropout(atom)
#         # atom1 = self.fc_g2(atom)  # torch.Size([256, 128])
#
#         # # bond
#         # bond = torch.unsqueeze(bond, 1)  # torch.Size([8655, 1, 132])
#         # bond = self.ugformer_layer1_1(bond)  # torch.Size([8655, 1, 132])
#         # bond = torch.squeeze(bond, 1)  # torch.Size([8655, 132])
#         # bond = self.fc_bond(bond)
#         # bond = self.relu(bond)
#         # bond = torch.unsqueeze(bond, 1)  # torch.Size([8655, 1, 132])
#         # bond = self.ugformer_layer2_2(bond)
#         # bond = torch.squeeze(bond, 1)
#         # bond = self.relu(bond)
#         # bond = torch.cat([gmp(bond, batch_bond), gap(bond, batch_bond)], dim=1)
#         # bond = self.relu(self.fc_g11(bond))
#         # bond = self.dropout(bond)
#         # bond1 = self.fc_g22(bond)
#
#         # smile
#         smile = self.smile2Vec(smiles_)
#         smile = self.fc_smile(smile)
#         for j in range(self.layer_smile):
#             smile = torch.relu(self.smile_W[j](smile))
#
#         # target_mut input feed-forward:
#         # target_mut = data.target_mut  # torch.Size([256, 735])
#         target_mut = target_mut[:, None, :]  # torch.Size([256, 1, 735])
#         conv_xt_mut = self.conv_xt_mut_1(target_mut)  # torch.Size([256, 32, 728])
#         conv_xt_mut = F.relu(conv_xt_mut)
#         conv_xt_mut = self.pool_xt_mut_1(conv_xt_mut)  # torch.Size([256, 32, 242])
#         conv_xt_mut = self.conv_xt_mut_2(conv_xt_mut)  # torch.Size([256, 64, 235])
#         conv_xt_mut = F.relu(conv_xt_mut)
#         conv_xt_mut = self.pool_xt_mut_2(conv_xt_mut)  # torch.Size([256, 64, 78])
#         conv_xt_mut = self.conv_xt_mut_3(conv_xt_mut)  # torch.Size([256, 128, 71])
#         conv_xt_mut = F.relu(conv_xt_mut)
#         conv_xt_mut = self.pool_xt_mut_3(conv_xt_mut)  # torch.Size([256, 128, 23])
#         xt_mut = conv_xt_mut.view(-1, conv_xt_mut.shape[1] * conv_xt_mut.shape[2])  # torch.Size([256, 2944])
#         xt_mut = self.fc1_xt_mut(xt_mut)  # torch.Size([256, 128])
#
#         # target_meth = data.target_meth
#         target_meth = target_meth[:, None, :]
#         conv_xt_meth = self.conv_xt_meth_1(target_meth)
#         conv_xt_meth = F.relu(conv_xt_meth)
#         conv_xt_meth = self.pool_xt_meth_1(conv_xt_meth)
#         conv_xt_meth = self.conv_xt_meth_2(conv_xt_meth)
#         conv_xt_meth = F.relu(conv_xt_meth)
#         conv_xt_meth = self.pool_xt_meth_2(conv_xt_meth)
#         conv_xt_meth = self.conv_xt_meth_3(conv_xt_meth)
#         conv_xt_meth = F.relu(conv_xt_meth)
#         conv_xt_meth = self.pool_xt_meth_3(conv_xt_meth)  # torch.Size([256, 128, 10])
#         xt_meth = conv_xt_meth.view(-1, conv_xt_meth.shape[1] * conv_xt_meth.shape[2])
#         xt_meth = self.fc1_xt_meth(xt_meth)
#
#         # target_ge = data.target_ge
#         target_ge = target_ge[:, None, :]
#         conv_xt_ge = self.conv_xt_ge_1(target_ge)
#         conv_xt_ge = F.relu(conv_xt_ge)
#         conv_xt_ge = self.pool_xt_ge_1(conv_xt_ge)
#         conv_xt_ge = self.conv_xt_ge_2(conv_xt_ge)
#         conv_xt_ge = F.relu(conv_xt_ge)
#         conv_xt_ge = self.pool_xt_ge_2(conv_xt_ge)
#         conv_xt_ge = self.conv_xt_ge_3(conv_xt_ge)
#         conv_xt_ge = F.relu(conv_xt_ge)
#         conv_xt_ge = self.pool_xt_ge_3(conv_xt_ge)
#         xt_ge = conv_xt_ge.view(-1, conv_xt_ge.shape[1] * conv_xt_ge.shape[2])
#         xt_ge = self.fc1_xt_ge(xt_ge)
#
#         # cat
#         xc = torch.cat((xt_mut, xt_meth, xt_ge, mol_vecs, smile), 1)  # bond1, atom1, smile, mol_vecs
#
#         # add some dense layers
#         xc = self.fc1(xc)
#         xc = self.relu(xc)
#         xc = self.batchnorm1(xc)
#         xc = self.dropout(xc)
#         xc = self.fc2(xc)
#         xc = self.relu1(xc)
#         xc = self.dropout(xc)
#         out = self.out(xc)
#         out = nn.Sigmoid()(out)
#         return out, atom, xc
#
#     def aggreate_to_atom_fea(self, message, a2x, atom_fea, linear_module):
#
#         a_message = self.select_neighbor_and_aggregate(message, a2x)
#         # do concat to atoms
#         a_input = torch.cat([atom_fea, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
#         atom_hiddens = self.act_func(linear_module(a_input))  # num_atoms x hidden
#         atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden
#
#         return atom_hiddens
#
#     def select_neighbor_and_aggregate(self, feature, index):  # 顶点的400维特征 每个顶点上所有的边序号信息
#         neighbor = index_select_ND(feature, index)
#         return neighbor.sum(dim=1)
import random

import networkx as nx
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GINConv, global_add_pool
from torch_geometric.nn import global_mean_pool as gap, global_max_pool as gmp
import math
import torch
from torch import nn
# from featurization import mol2graph, index_select_ND, BatchGRU
from smile2Vec import smile2Vec

class ModDRDSP(torch.nn.Module):
    def __init__(self, n_output=1, num_features_xd=119, num_features_xs=132,
                 n_filters=32, layer_smile=3, hidden_dim=1500, output_dim=128, dropout=0.2):

        super(ModDRDSP, self).__init__()

        # # atom
        # self.num_features_xd = num_features_xd
        # self.encoder_layer_1 = nn.TransformerEncoderLayer(d_model=num_features_xd, nhead=1, dropout=0.5)
        # self.ugformer_layer_1 = nn.TransformerEncoder(self.encoder_layer_1, 1)
        # self.conv1 = GATConv(num_features_xd, num_features_xd, heads=10)
        # self.relu = nn.ReLU()
        # self.encoder_layer_2 = nn.TransformerEncoderLayer(d_model=num_features_xd * 10, nhead=1, dropout=0.5)
        # self.ugformer_layer_2 = nn.TransformerEncoder(self.encoder_layer_2, 1)
        # self.conv2 = GCNConv(num_features_xd * 10, num_features_xd * 10)
        # self.fc_g1 = torch.nn.Linear(num_features_xd * 10 * 2, hidden_dim)
        # self.fc_g2 = torch.nn.Linear(hidden_dim, output_dim)
        # self.dropout = nn.Dropout(dropout)
        #
        # # bond
        # self.encoder_layer1_1 = nn.TransformerEncoderLayer(d_model=num_features_xs, nhead=1, dropout=0.5)
        # self.ugformer_layer1_1 = nn.TransformerEncoder(self.encoder_layer1_1, 1)
        # self.fc_bond = nn.Linear(num_features_xs, num_features_xs * 10)
        # self.relu = nn.ReLU()
        # self.encoder_layer2_2 = nn.TransformerEncoderLayer(d_model=num_features_xs * 10, nhead=1, dropout=0.5)
        # self.ugformer_layer2_2 = nn.TransformerEncoder(self.encoder_layer2_2, 1)
        # self.fc_g11 = torch.nn.Linear(num_features_xs * 2 * 10, hidden_dim)
        # self.fc_g22 = torch.nn.Linear(hidden_dim, output_dim)

        # cell line mut feature
        self.conv_xt_mut_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_mut_1 = nn.MaxPool1d(3)
        self.conv_xt_mut_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_mut_2 = nn.MaxPool1d(3)
        self.conv_xt_mut_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt_mut_3 = nn.MaxPool1d(3)
        self.fc1_xt_mut = nn.Linear(2944, output_dim)

        # cell line meth feature
        self.conv_xt_meth_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_meth_1 = nn.MaxPool1d(3)
        self.conv_xt_meth_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_meth_2 = nn.MaxPool1d(3)
        self.conv_xt_meth_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt_meth_3 = nn.MaxPool1d(3)
        self.fc1_xt_meth = nn.Linear(1280, output_dim)

        # cell line ge feature
        self.conv_xt_ge_1 = nn.Conv1d(in_channels=1, out_channels=n_filters, kernel_size=8)
        self.pool_xt_ge_1 = nn.MaxPool1d(3)
        self.conv_xt_ge_2 = nn.Conv1d(in_channels=n_filters, out_channels=n_filters * 2, kernel_size=8)
        self.pool_xt_ge_2 = nn.MaxPool1d(3)
        self.conv_xt_ge_3 = nn.Conv1d(in_channels=n_filters * 2, out_channels=n_filters * 4, kernel_size=8)
        self.pool_xt_ge_3 = nn.MaxPool1d(3)
        self.fc1_xt_ge = nn.Linear(4224, output_dim)

        # mlp
        self.conv_cell = nn.Sequential(
            nn.Linear(output_dim*3, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(output_dim, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )

        # smile2Vec GRU
        self.smile2Vec = smile2Vec(np.load("data/smi_embedding_matrix.npy"))
        self.layer_smile = layer_smile
        self.fc_smile = nn.Linear(150+128*3, output_dim)
        self.smile_W = nn.ModuleList([nn.Linear(output_dim, output_dim)
                                      for _ in range(layer_smile)])


        # Input
        atom_fdim = num_features_xd  # 133
        self.hidden_size = 300
        self.bias = False
        self.dropout_layer = nn.Dropout(p=0.1)
        self.W_i_atom = nn.Linear(atom_fdim, self.hidden_size, bias=self.bias)
        bond_fdim = num_features_xs  # 147
        self.W_i_bond = nn.Linear(bond_fdim, self.hidden_size, bias=self.bias)
        self.act_func = nn.ReLU()
        w_h_input_size_atom = self.hidden_size + bond_fdim

        self.W_h_atom = nn.Linear(w_h_input_size_atom, self.hidden_size, bias=self.bias)

        w_h_input_size_bond = self.hidden_size

        self.depth = 2
        for depth in range(self.depth):
            self._modules[f'W_h_{depth}'] = nn.Linear(w_h_input_size_bond, self.hidden_size, bias=self.bias)

        self.W_o = nn.Linear(self.hidden_size * 4, self.hidden_size*2)
        # not share parameter between node and edge so far
        self.W_nout = nn.Linear(atom_fdim + self.hidden_size, self.hidden_size)
        self.W_eout = nn.Linear(atom_fdim + self.hidden_size, self.hidden_size)

        self.gru = BatchGRU(self.hidden_size*2)

        self.lr = nn.Linear(self.hidden_size * 4 + atom_fdim, self.hidden_size*2, bias=self.bias)

        # combined layers
        self.fc1 = nn.Linear(4 * output_dim + 600, 1024)
        self.batchnorm1 = nn.BatchNorm1d(1024)
        self.relu1 = nn.LeakyReLU()
        self.fc2 = nn.Linear(1024, 256)
        self.out = nn.Linear(256, n_output)

    def forward(self, atom, bond, edge_index, a2a, a2b, a_scope, b_scope, b2a, b2revb, \
                batch_atom, batch_bond, smiles_, smiles, target_mut, target_meth, target_ge):

        # target_mut input feed-forward: # torch.Size([256, 735])
        target_mut = target_mut[:, None, :]  # torch.Size([256, 1, 735])
        conv_xt_mut = self.conv_xt_mut_1(target_mut)  # torch.Size([256, 32, 728])
        conv_xt_mut = F.relu(conv_xt_mut)
        conv_xt_mut = self.pool_xt_mut_1(conv_xt_mut)  # torch.Size([256, 32, 242])
        conv_xt_mut = self.conv_xt_mut_2(conv_xt_mut)  # torch.Size([256, 64, 235])
        conv_xt_mut = F.relu(conv_xt_mut)
        conv_xt_mut = self.pool_xt_mut_2(conv_xt_mut)  # torch.Size([256, 64, 78])
        conv_xt_mut = self.conv_xt_mut_3(conv_xt_mut)  # torch.Size([256, 128, 71])
        conv_xt_mut = F.relu(conv_xt_mut)
        conv_xt_mut = self.pool_xt_mut_3(conv_xt_mut)  # torch.Size([256, 128, 23])
        xt_mut = conv_xt_mut.view(-1, conv_xt_mut.shape[1] * conv_xt_mut.shape[2])  # torch.Size([256, 2944])
        xt_mut = self.fc1_xt_mut(xt_mut)  # torch.Size([256, 128])

        # target_meth = data.target_meth
        target_meth = target_meth[:, None, :]
        conv_xt_meth = self.conv_xt_meth_1(target_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_1(conv_xt_meth)
        conv_xt_meth = self.conv_xt_meth_2(conv_xt_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_2(conv_xt_meth)
        conv_xt_meth = self.conv_xt_meth_3(conv_xt_meth)
        conv_xt_meth = F.relu(conv_xt_meth)
        conv_xt_meth = self.pool_xt_meth_3(conv_xt_meth)  # torch.Size([256, 128, 10])
        xt_meth = conv_xt_meth.view(-1, conv_xt_meth.shape[1] * conv_xt_meth.shape[2])
        xt_meth = self.fc1_xt_meth(xt_meth)

        # target_ge = data.target_ge
        target_ge = target_ge[:, None, :]
        conv_xt_ge = self.conv_xt_ge_1(target_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_1(conv_xt_ge)
        conv_xt_ge = self.conv_xt_ge_2(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_2(conv_xt_ge)
        conv_xt_ge = self.conv_xt_ge_3(conv_xt_ge)
        conv_xt_ge = F.relu(conv_xt_ge)
        conv_xt_ge = self.pool_xt_ge_3(conv_xt_ge)
        xt_ge = conv_xt_ge.view(-1, conv_xt_ge.shape[1] * conv_xt_ge.shape[2])
        xt_ge = self.fc1_xt_ge(xt_ge)

        # cat
        cell = torch.cat((xt_mut, xt_meth, xt_ge), 1)  # 128*3
        # cell = self.conv_cell(cell)  # 128

        # atom_ = atom
        # bond_ = bond
        # Transformer_atom
        # atom = torch.unsqueeze(atom, 1)  # torch.Size([4173, 119])-->torch.Size([4173, 1, 119])
        # atom = self.ugformer_layer_1(atom)  # torch.Size([4173, 1, 119])
        # atom = torch.squeeze(atom, 1)  # torch.Size([4173, 119])

        # cell_a = torch.zeros(atom.shape[0], cell.size(1))  # 创建一个全零的张量
        # start_idx = 1
        # index = 0
        # for a_start, a_size in a_scope:
        #     cell_a[start_idx:start_idx + a_size, :] = cell[index]
        #     start_idx += a_size
        #     index += 1
        #
        # cell_b = torch.zeros(bond.shape[0], cell.size(1))  # 创建一个全零的张量
        # start_idx = 1
        # index = 0
        # for a_start, a_size in b_scope:
        #     cell_b[start_idx:start_idx + a_size, :] = cell[index]
        #     start_idx += a_size
        #     index += 1

        # atom = torch.cat((atom, cell_a.cuda()), 1)
        # bond = torch.cat((bond, cell_b.cuda()), 1)

        # Transformer_bond
        # bond = torch.unsqueeze(bond, 1)  # torch.Size([8655, 1, 132])
        # bond = self.ugformer_layer1_1(bond)  # torch.Size([8655, 1, 132])
        # bond = torch.squeeze(bond, 1)  # torch.Size([8655, 132])

        # get_edges
        # atom_adj, bond_adj = get_edges(atom, bond, a2a, a2b)

        # Input 首先输入网络后先进行了线性层
        input_atom = self.W_i_atom(atom)  # torch.Size([4173, 300])
        input_atom = self.act_func(input_atom)  # torch.Size([4173, 300])
        message_atom = input_atom.clone()  # torch.Size([4173, 300])

        input_bond = self.W_i_bond(bond)  # torch.Size([9129, 132]) -->torch.Size([9129, 300])
        input_bond = self.act_func(input_bond)  # torch.Size([9129, 300])
        message_bond = input_bond.clone()  # torch.Size([9129, 300])

        # message_bond = torch.cat((message_bond, input_bond), dim=1)

        # 使用消息聚合模块聚集临近节点的信息
        nei_message = self.select_neighbor_and_aggregate(message_atom, a2a)  # 根据边聚合  torch.Size([4173, 300])  torch.Size([4173, 4])--?torch.Size([4173, 300])
        nei_attached_message = self.select_neighbor_and_aggregate(input_bond, a2b)  # 根据边聚合  torch.Size([9129, 300])  torch.Size([4173, 4])--?torch.Size([4173, 300])
        # atom_message = torch.cat((nei_message, nei_attached_message), dim=1)  # torch.Size([4173, 600])
        atom_message = nei_message + nei_attached_message
        # Message passing
        for depth in range(self.depth):  # 首先原子的特征维度为[1566, 133]   边的特征维度为 [3347, 147]， 首先得到每个原子上面的边以及维度，然后将得到的新特征加到原来的原子特征上
            agg_message = index_select_ND(message_bond, a2b)  # torch.Size([4173, 4, 300])
            agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]  # torch.Size([4173, 300])*torch.Size([4173, 300])->torch.Size([4173, 300])
            message_atom = atom_message + agg_message  # torch.Size([4173, 300])

            # directed graph
            rev_message = input_bond[b2revb]  # torch.Size([9129, 300])
            # rev_message = torch.cat((rev_message, input_bond[b2a[b2revb]]), dim=1)
            rev_message = rev_message + input_bond[b2a[b2revb]]
            message_bond = message_atom[b2a] - rev_message  # torch.Size([9129, 300])

            message_bond = self._modules[f'W_h_{depth}'](message_bond)  # liner torch.Size([9129, 300])
            message_bond = self.dropout_layer(self.act_func(message_bond))

        agg_message = index_select_ND(message_bond, a2b)  # torch.Size([4173, 4, 300])
        agg_message = agg_message.sum(dim=1) * agg_message.max(dim=1)[0]  # torch.Size([4173, 300])
        message_atom = message_atom + agg_message

        # 交叉传递
        message_atom = self.aggreate_to_atom_fea(message_atom, a2a, atom,
                                                 linear_module=self.W_nout)  # torch.Size([4173, 300])
        message_bond = self.aggreate_to_atom_fea(message_bond, a2b, atom,
                                                 linear_module=self.W_eout)  # torch.Size([4173, 300])

        agg_message = self.lr(torch.cat([agg_message, message_atom, message_bond, atom_message, atom], 1))  # torch.Size([4173, 300])

        agg_message = self.gru(agg_message, a_scope)  # torch.Size([4173, 600])

        atom_hiddens = self.dropout_layer(self.act_func(self.W_o(agg_message)))  # torch.Size([4173, 300])

        # Readout
        mol_vecs = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_hiddens = atom_hiddens.narrow(0, a_start, a_size)
            mol_vecs.append(cur_hiddens.mean(0))
        mol_vecs = torch.stack(mol_vecs, dim=0)  # torch.Size([128, 300])

        # # atom
        # atom = torch.unsqueeze(atom, 1)
        # atom = self.ugformer_layer_1(atom)
        # atom = torch.squeeze(atom, 1)
        # # residual = self.linear(atom)
        # atom = self.conv1(atom, edge_index)
        # atom = self.relu(atom)
        # # atom = atom + residual
        # atom = torch.unsqueeze(atom, 1)
        # atom = self.ugformer_layer_2(atom)
        # atom = torch.squeeze(atom, 1)
        # # residual = self.linear2(atom)
        # atom = self.conv2(atom, edge_index)
        # atom = self.relu(atom)
        # # atom = atom + residual
        # atom = torch.cat([gmp(atom, batch_atom), gap(atom, batch_atom)],
        #                  dim=1)  # apply global max pooling (gmp) and global mean pooling (gap)
        # atom = self.relu(self.fc_g1(atom))
        # atom = self.dropout(atom)
        # atom1 = self.fc_g2(atom)  # torch.Size([256, 128])

        # # bond
        # bond = torch.unsqueeze(bond, 1)  # torch.Size([8655, 1, 132])
        # bond = self.ugformer_layer1_1(bond)  # torch.Size([8655, 1, 132])
        # bond = torch.squeeze(bond, 1)  # torch.Size([8655, 132])
        # bond = self.fc_bond(bond)
        # bond = self.relu(bond)
        # bond = torch.unsqueeze(bond, 1)  # torch.Size([8655, 1, 132])
        # bond = self.ugformer_layer2_2(bond)
        # bond = torch.squeeze(bond, 1)
        # bond = self.relu(bond)
        # bond = torch.cat([gmp(bond, batch_bond), gap(bond, batch_bond)], dim=1)
        # bond = self.relu(self.fc_g11(bond))
        # bond = self.dropout(bond)
        # bond1 = self.fc_g22(bond)

        # smile
        smile = self.smile2Vec(smiles_)
        # smile = torch.cat((smile, cell), 1)
        smile = self.fc_smile(smile)
        for j in range(self.layer_smile):
            smile = torch.relu(self.smile_W[j](smile))


        # cat
        xc = torch.cat((cell, mol_vecs, smile), 1)  # bond1, atom1, smile, mol_vecs
        mol = torch.cat((xt_mut, xt_meth, xt_ge, mol_vecs), 1)
        # add some dense layers
        xc = self.fc1(xc)
        xc = self.relu(xc)
        xc = self.batchnorm1(xc)
        xc = self.dropout(xc)
        xc = self.fc2(xc)
        xc = self.relu1(xc)
        xc = self.dropout(xc)
        out = self.out(xc)
        out = nn.Sigmoid()(out)
        return out, mol, smile, cell

    def aggreate_to_atom_fea(self, message, a2x, atom_fea, linear_module):

        a_message = self.select_neighbor_and_aggregate(message, a2x)
        # do concat to atoms
        a_input = torch.cat([atom_fea, a_message], dim=1)  # num_atoms x (atom_fdim + hidden)
        atom_hiddens = self.act_func(linear_module(a_input))  # num_atoms x hidden
        atom_hiddens = self.dropout_layer(atom_hiddens)  # num_atoms x hidden

        return atom_hiddens

    def select_neighbor_and_aggregate(self, feature, index):  # 顶点的400维特征 每个顶点上所有的边序号信息
        neighbor = index_select_ND(feature, index)
        return neighbor.sum(dim=1)

def index_select_ND(source: torch.Tensor, index: torch.Tensor) -> torch.Tensor:
    """
    Selects the message features from source corresponding to the atom or bond indices in index.

    :param source: A tensor of shape (num_bonds, hidden_size) containing message features.
    :param index: A tensor of shape (num_atoms/num_bonds, max_num_bonds) containing the atom or bond
    indices to select from source.
    :return: A tensor of shape (num_atoms/num_bonds, max_num_bonds, hidden_size) containing the message
    features corresponding to the atoms/bonds specified in index.
    """
    index_size = index.size()  # (num_atoms/num_bonds, max_num_bonds)
    suffix_dim = source.size()[1:]  # (hidden_size,)
    final_size = index_size + suffix_dim  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target = source.index_select(dim=0, index=index.view(-1))  # (num_atoms/num_bonds * max_num_bonds, hidden_size)
    target = target.view(final_size)  # (num_atoms/num_bonds, max_num_bonds, hidden_size)

    target[index == 0] = 0
    return target

class BatchGRU(nn.Module):
    def __init__(self, hidden_size=300):
        super(BatchGRU, self).__init__()
        self.hidden_size = hidden_size
        self.gru = nn.GRU(self.hidden_size, self.hidden_size, batch_first=True,
                          bidirectional=True)
        self.bias = nn.Parameter(torch.Tensor(self.hidden_size))
        self.bias.data.uniform_(-1.0 / math.sqrt(self.hidden_size),
                                1.0 / math.sqrt(self.hidden_size))

    def forward(self, node, a_scope):
        hidden = node
        message = F.relu(node + self.bias)
        MAX_atom_len = max([a_size for a_start, a_size in a_scope])
        # padding
        message_lst = []
        hidden_lst = []
        for i, (a_start, a_size) in enumerate(a_scope):
            if a_size == 0:
                assert 0
            cur_message = message.narrow(0, a_start, a_size)
            cur_hidden = hidden.narrow(0, a_start, a_size)
            hidden_lst.append(cur_hidden.max(0)[0].unsqueeze(0).unsqueeze(0))

            cur_message = torch.nn.ZeroPad2d((0, 0, 0, MAX_atom_len - cur_message.shape[0]))(cur_message)
            message_lst.append(cur_message.unsqueeze(0))

        message_lst = torch.cat(message_lst, 0)
        hidden_lst = torch.cat(hidden_lst, 1)
        hidden_lst = hidden_lst.repeat(2, 1, 1)
        cur_message, cur_hidden = self.gru(message_lst, hidden_lst)

        # unpadding
        cur_message_unpadding = []
        for i, (a_start, a_size) in enumerate(a_scope):
            cur_message_unpadding.append(cur_message[i, :a_size].view(-1, 2 * self.hidden_size))
        cur_message_unpadding = torch.cat(cur_message_unpadding, 0)

        message = torch.cat([torch.cat([message.narrow(0, 0, 1), message.narrow(0, 0, 1)], 1),
                             cur_message_unpadding], 0)
        return message
