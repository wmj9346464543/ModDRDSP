import numpy as np
import pandas as pd
import sys, os
from random import shuffle
import torch
import torch.nn as nn
from torch_geometric.data import InMemoryDataset, DataLoader
from utils import *
import argparse
import pickle
from sklearn.ensemble import RandomForestRegressor
from data_process import TestbedDataset
from model import ModDRDSP
Print = True

def graph_process(data):
    atom, bond, edge_index2, c_size, b_size, edge_index1, batch_atom, smiles_seq, smiles, target_mut, \
    target_meth, target_ge, y = data.x, data.xc, data.edge_index2, data.c_size, data.b_size, \
                                data.edge_index1, data.batch, data.smiles_seq, data.smiles, data.target_mut, \
                                data.target_meth, data.target_ge, data.y
    # 1.获取f_atoms
    # atom
    atom = torch.cat((torch.zeros((1, atom.shape[1])), atom), dim=0)  # , device='cuda:0'
    # 2.获取f_bonds
    # bond
    bond = torch.repeat_interleave(bond, 2, dim=0)
    bond = torch.cat((torch.zeros((1, bond.shape[1])), bond), dim=0)  # , device='cuda:0'
    # 3.获取bonds
    # edge_index
    edge_index = edge_index2 + 1
    edge_index = torch.cat((torch.zeros((edge_index.shape[0], 1), dtype=int), edge_index), dim=1)  # , device='cuda:0'

    # 4.获取a2b
    a2b = torch.zeros((atom.shape[0], 4), dtype=int)  # , device='cuda:0'
    deg = torch.zeros(atom.shape[0], dtype=torch.long)
    for i, (r, _) in enumerate(edge_index.t()):
        a2b[r, deg[r]] = i
        deg[r] += 1
    # 5.获取a_scope
    a_scope = []
    start = 1
    for size in c_size:
        a_scope.append((start, int(size)))
        start = start + int(size)
    # 6.获取b_scope
    b_scope = []
    start = 1
    for size in b_size:
        b_scope.append((start, int(size) * 2))
        start = start + int(size) * 2

    # 7.获取b2a
    b2a = torch.cat((torch.zeros(1, dtype=int), edge_index1.t().reshape(-1) + 1))  # , device='cuda:0'
    # 8.获取b2revb
    b2revb = torch.cat((torch.tensor([0]),  # , device='cuda:0'
                        torch.tensor([(i + 1, i) for i in range(1, bond.shape[0], 2)]).reshape(  # , device='cuda:0'
                            -1)))
    # 9.获取
    batch_atom = torch.cat((torch.zeros(1, dtype=int), batch_atom))  # , device='cuda:0'

    # 10.
    batch_bond, index, n = [0] * bond.shape[0], 0, 0  # 创建一个初始为零的列表，长度为所有分子的总原子数
    for b in b_size:
        for _ in range(b):
            batch_bond[index] = n  # 在 batch 列表中，将当前索引位置设置为 1
            index += 1  # 索引递增，指向下一个位置
        n += 1
    batch_bond = torch.LongTensor(batch_bond)  # .to('cuda')

    # 11. a2a
    a2a = b2a[a2b]
    return atom.cuda(), bond.cuda(), edge_index.cuda(), a2a.cuda(), a2b.cuda(), a_scope, b_scope, b2a.cuda(), b2revb.cuda(), \
           batch_atom.cuda(), batch_bond.cuda(), smiles_seq.cuda(), smiles, target_mut.cuda(), target_meth.cuda(), target_ge.cuda(), y.cuda()

def train(model, device, train_loader, optimizer, epoch, log_interval):
    model.train()
    loss_fn = nn.MSELoss()
    avg_loss = []
    score = torch.Tensor().cuda()
    score2 = torch.Tensor().cuda()

    ##
    # data_frame = pd.read_csv('data/traintrain.csv', header=None)
    # excel_data = data_frame.values
    # tensor_data = torch.from_numpy(excel_data)
    # num_elements = tensor_data.shape[0]
    # batch_size = 256
    # elements_per_batch = num_elements // batch_size
    # new_y = tensor_data[:batch_size * elements_per_batch].view(elements_per_batch, batch_size)
    # last_batch_size = num_elements % batch_size
    # last_batch_idx = elements_per_batch
    # last_batch = tensor_data[batch_size * last_batch_idx:].squeeze()
    ##
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_mol = torch.Tensor()
    total_smile = torch.Tensor()
    total_cell = torch.Tensor()

    if not os.path.exists('preprocessed_data.pkl'):
        preprocessed_data = []
        for batch_idx, data in enumerate(train_loader):
            print(batch_idx)
            preprocessed_data.append((graph_process(data)))
        with open('preprocessed_data.pkl', 'wb') as f:
            pickle.dump(preprocessed_data, f)
        # exit(0)
        # if os.path.exists('./data/processed/train_gragh_list/{}.pkl'.format(str(batch_idx))):
        #     with open('./data/processed/train_gragh_list/{}.pkl'.format(str(batch_idx)), 'rb') as file:
        #         _, gragh = pickle.load(file)
        # else:
        #     with open('./data/processed/train_gragh_list/{}.pkl'.format(str(batch_idx)), 'wb') as file:
        #         gragh = dgl.batch(Dataset(data.smiles))
        #         pickle.dump((data, gragh), file)
    with open('preprocessed_data.pkl', 'rb') as f:
        del train_loader
        preprocessed_data = pickle.load(f)
    for batch_idx, data in enumerate(preprocessed_data):
        atom, bond, edge_index, a2a, a2b, a_scope, b_scope, b2a, b2revb, \
        batch_atom, batch_bond, smiles_, smiles, target_mut, target_meth, target_ge, y = data

        atom, bond, edge_index, a2a, a2b, b2a, b2revb, \
        batch_atom, batch_bond, smiles_, target_mut, target_meth, target_ge, y = \
            atom.cuda(), bond.cuda(), edge_index.cuda(), a2a.cuda(), \
            a2b.cuda(), b2a.cuda(), b2revb.cuda(), batch_atom.cuda(), \
            batch_bond.cuda(), smiles_.cuda(), target_mut.cuda(), target_meth.cuda(), target_ge.cuda(), y.cuda()

        optimizer.zero_grad()
        output, mol, smile, cell = model(atom, bond, edge_index, a2a, a2b, a_scope, b_scope, b2a, b2revb, \
                                         batch_atom, batch_bond, smiles_, smiles, target_mut, target_meth, target_ge)
        # total_mol = torch.cat((total_mol, mol.detach().cpu()), 0)
        # total_smile = torch.cat((total_smile, smile.detach().cpu()), 0)
        # total_cell = torch.cat((total_cell, cell.detach().cpu()), 0)
        # score = torch.cat((score, y.view(-1, 1).float().to(device)), 0)
        # score1 = torch.cat((output, y.view(-1, 1).float().to(device)), 1)
        # score2 = torch.cat((score2, score1), 0)
        # if epoch == 10:
        #     print(epoch)
        #     import umap
        #     from sklearn.datasets import load_digits
        #     import matplotlib.pyplot as plt
        #     from matplotlib.font_manager import FontProperties
        #     import numpy as np
        #     reducer = umap.UMAP(n_jobs=2)
        #     embedding = reducer.fit_transform(smile.detach().cpu())
        #     plt.scatter(embedding[:, 0], embedding[:, 1], c=y.view(-1, 1).float().to('cpu'), cmap='Spectral', s=20)
        #     plt.gca().set_aspect('equal', 'datalim')
        #     plt.colorbar(boundaries=np.arange(3) - 0.5).set_ticks(np.arange(2))  # 注意这里的boundaries和ticks需要根据类别数目进行调整
        #     plt.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        #     plt.tick_params(axis='y', which='both', left=False, right=False, labelleft=False)
        #     plt.show()

        loss = loss_fn(output, y.view(-1, 1).float().to(device))
        loss.backward()
        optimizer.step()
        avg_loss.append(loss.item())
        if batch_idx % log_interval == 0:
            print('Train epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(epoch,
                                                                           batch_idx * len(atom),
                                                                           len(preprocessed_data),
                                                                           100. * batch_idx / len(preprocessed_data),
                                                                           loss.item()))
            # 将数据保存为csv文件

        total_preds = torch.cat((total_preds, output.detach().cpu()), 0)
        total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)
    # np.savetxt("data/" + "traintrain.csv", score2.cpu().detach().numpy())  # 118158, 128)

    return sum(avg_loss) / len(avg_loss), total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_mol, total_smile, total_cell


def predicting(model, device, loader, a):
    if a == 2:
        name = "test"
    else:
        name = "val"
    model.eval()
    avg_loss = []
    loss_fn = nn.MSELoss()
    total_preds = torch.Tensor()
    total_labels = torch.Tensor()
    total_mol = torch.Tensor()
    total_smile = torch.Tensor()
    total_cell = torch.Tensor()
    print('Make prediction for ')  # {} samples...'.format(len(loader.dataset)))
    with torch.no_grad():
        score = torch.Tensor().cuda()
        score2 = torch.Tensor().cuda()
        if not os.path.exists('preprocessed_data_{}.pkl'.format(name)):
            preprocessed_data = []
            for batch_idx, data in enumerate(loader):
                print(batch_idx)
                preprocessed_data.append((graph_process(data)))
            with open('preprocessed_data_{}.pkl'.format(name), 'wb') as f:
                pickle.dump(preprocessed_data, f)
        with open('preprocessed_data_{}.pkl'.format(name), 'rb') as f:
            preprocessed_data = pickle.load(f)
        for batch_idx, data in enumerate(preprocessed_data):
            atom, bond, edge_index, a2a, a2b, a_scope, b_scope, b2a, b2revb, \
            batch_atom, batch_bond, smiles_, smiles, target_mut, target_meth, target_ge, y = data

            atom, bond, edge_index, a2a, a2b, b2a, b2revb, \
            batch_atom, batch_bond, smiles_, target_mut, target_meth, target_ge, y = \
                atom.cuda(), bond.cuda(), edge_index.cuda(), a2a.cuda(), \
                a2b.cuda(), b2a.cuda(), b2revb.cuda(), batch_atom.cuda(), \
                batch_bond.cuda(), smiles_.cuda(), target_mut.cuda(), target_meth.cuda(), target_ge.cuda(), y.cuda()

            # for batch_idx, data in enumerate(loader):
            # ##
            # data_frame = pd.read_csv("data/" + name + "ttt.csv", header=None)
            # excel_data = data_frame.values
            # tensor_data = torch.from_numpy(excel_data)
            # num_elements = tensor_data.shape[0]
            # batch_size = 256
            # elements_per_batch = num_elements // batch_size
            # new_y = tensor_data[:batch_size * elements_per_batch].view(elements_per_batch, batch_size)
            # last_batch_size = num_elements % batch_size
            # last_batch_idx = elements_per_batch
            # last_batch = tensor_data[batch_size * last_batch_idx:].squeeze()
            ##
            ##
            # if os.path.exists('./data/processed/{}_gragh_list/{}.pkl'.format(name, str(batch_idx))):
            #     with open('./data/processed/{}_gragh_list/{}.pkl'.format(name, str(batch_idx)), 'rb') as file:
            #         _, gragh = pickle.load(file)
            # else:
            #     with open('./data/processed/{}_gragh_list/{}.pkl'.format(name, str(batch_idx)), 'wb') as file:
            #         gragh = dgl.batch(Dataset(data.smiles))
            #         pickle.dump((data, gragh), file)
            # data = data.to(device)
            # gragh = gragh.to(device)

            # if y.shape[0] != batch_size:
            #     y = last_batch
            # else:
            #     y = new_y[batch_idx]
            # output, _, xc = model(data)

            output, mol, smile, cell = model(atom, bond, edge_index, a2a, a2b, a_scope, b_scope, b2a, b2revb, \
                                             batch_atom, batch_bond, smiles_, smiles, target_mut, target_meth,
                                             target_ge)

            # total_mol = torch.cat((total_mol, mol.detach().cpu()), 0)
            # total_smile = torch.cat((total_smile, smile.detach().cpu()), 0)
            # total_cell = torch.cat((total_cell, cell.detach().cpu()), 0)
            # score = torch.cat((score, y.view(-1, 1).float().to(device)), 0)
            # # print(y.view(-1, 1).float().to(device))
            # # xc = xc.squeeze()
            # score1 = torch.cat((output, y.view(-1, 1).float().to(device)), 1)
            # score2 = torch.cat((score2, score1), 0)
            # # print(score)
            loss = loss_fn(output, y.view(-1, 1).float().to(device))
            avg_loss.append(loss.item())
            total_preds = torch.cat((total_preds, output.cpu()), 0)
            total_labels = torch.cat((total_labels, y.view(-1, 1).cpu()), 0)
        # np.savetxt("data/" + name + "ttt.csv", score2.cpu().detach().numpy())
    # del preprocessed_data
    return sum(avg_loss) / len(avg_loss), total_labels.numpy().flatten(), total_preds.numpy().flatten(), total_mol, total_smile, total_cell


def main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name):
    print('Learning rate: ', lr)
    print('Epochs: ', num_epoch)
    dataset = 'GDSC'
    train_losses = []
    val_losses = []
    val_pearsons = []
    processed_data_file_train = 'data/processed/' + dataset + '_train_mix' + '.pt'
    processed_data_file_val = 'data/processed/' + dataset + '_val_mix' + '.pt'
    processed_data_file_test = 'data/processed/' + dataset + '_test_mix' + '.pt'
    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_val)) or (
            not os.path.isfile(processed_data_file_test))):
        print('please run data_process.py to prepare data in pytorch format!')
    else:

        if not os.path.exists('preprocessed_data.pkl'):
            train_data = TestbedDataset(root='data', dataset=dataset + '_train_mix')
            val_data = TestbedDataset(root='data', dataset=dataset + '_val_mix')
            test_data = TestbedDataset(root='data', dataset=dataset + '_test_mix')
            print(1)
            train_loader = DataLoader(train_data, batch_size=train_batch, shuffle=False)
            val_loader = DataLoader(val_data, batch_size=val_batch, shuffle=False)
            test_loader = DataLoader(test_data, batch_size=test_batch, shuffle=False)
            print(11)

            # preprocessed_data = []
            # for batch_idx, data in enumerate(train_loader):
            #     print(batch_idx)
            #     preprocessed_data.append((graph_process(data)))
            # with open('preprocessed_data.pkl', 'wb') as f:
            #     pickle.dump(preprocessed_data, f)
            #     print(111)
            #
            # preprocessed_data = []
            # for batch_idx, data in enumerate(val_loader):
            #     print(batch_idx)
            #     preprocessed_data.append((graph_process(data)))
            # with open('preprocessed_data_{}.pkl'.format("val"), 'wb') as f:
            #     pickle.dump(preprocessed_data, f)
            #     print(111)
            #
            # preprocessed_data = []
            # for batch_idx, data in enumerate(test_loader):
            #     print(batch_idx)
            #     preprocessed_data.append((graph_process(data)))
            # with open('preprocessed_data_{}.pkl'.format("test"), 'wb') as f:
            #     pickle.dump(preprocessed_data, f)
            #     print(111)
        else:
            train_loader = val_loader = test_loader =[]
        # training the model

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=15, min_lr=0.00001)
        best_mse = 1000
        best_pearson = 1
        best_epoch = -1
        model_file_name = 'model_' + '_' + dataset + '.model'
        result_file_name = 'result_' + '_' + dataset + '.csv'
        loss_fig_name = 'model_' + '_' + dataset + '_loss'
        pearson_fig_name = 'model_' + '_' + dataset + '_pearson'
        for epoch in range(num_epoch):
            train_loss, G, P, total_mol, total_smile, total_cell = train(model, device, train_loader, optimizer, epoch + 1, log_interval)
            # combined_features = np.concatenate((torch.tensor(P).unsqueeze(1), total_mol, total_smile, total_cell), axis=1)
            ret = [rmse(G, P), mse(G, P), pearson(G, P), spearman(G, P)]

            val_loss, G_v, P_v, total_mol, total_smile, total_cell = predicting(model, device, val_loader, a=1)
            # combined_features_v = np.concatenate((torch.tensor(P_v).unsqueeze(1), total_mol, total_smile, total_cell), axis=1)
            ret_v = [rmse(G_v, P_v), mse(G_v, P_v), pearson(G_v, P_v), spearman(G_v, P_v)]

            test_loss, G_t, P_t, total_mol, total_smile, total_cell = predicting(model, device, test_loader, a=2)
            # combined_features_t = np.concatenate((torch.tensor(P_t).unsqueeze(1), total_mol, total_smile, total_cell), axis=1)
            ret_t = [rmse(G_t, P_t), mse(G_t, P_t), pearson(G_t, P_t), spearman(G_t, P_t)]

            # if epoch == 30 or epoch == 50 or epoch == 90 or epoch == 0:
            #     np.save('combined_features{}.npy'.format(epoch), np.array(combined_features))
            #     np.save('combined_features_v{}.npy'.format(epoch), np.array(combined_features_v))
            #     np.save('combined_features_t{}.npy'.format(epoch), np.array(combined_features_t))
            #     np.save('G{}.npy'.format(epoch), np.array(G))
            #     np.save('G_v{}.npy'.format(epoch), np.array(G_v))
            #     np.save('G_t{}.npy'.format(epoch), np.array(G_t))

            if Print:
                print("Train Loss:", train_loss)
                print("Train RMSE:", ret[0])
                print("Train MSE:", ret[1])
                print("Train Pearson:", ret[2])
                print("Train Spearman:", ret[3])
                print("Validation Loss:", val_loss)
                print("Validation RMSE:", ret_v[0])
                print("Validation MSE:", ret_v[1])
                print("Validation Pearson:", ret_v[2])
                print("Validation Spearman:", ret_v[3])
                print("Test Loss:", test_loss)
                print("Test RMSE:", ret_t[0])
                print("Test MSE:", ret_t[1])
                print("Test Pearson:", ret_t[2])
                print("Test Spearman:", ret_t[3])
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_pearsons.append(ret_v[2])

            # Reduce Learning rate on Plateau for the validation loss
            scheduler.step(val_loss)
            print("当前学习率：", optimizer.param_groups[0]['lr'])
            if ret_v[1] < best_mse:
                torch.save(model.state_dict(), model_file_name)
                with open(result_file_name, 'w') as f:
                    f.write(','.join(map(str, ret_t)))
                best_epoch = epoch + 1
                best_mse = ret_v[1]
                best_pearson = ret_v[2]
                print(' rmse improved at epoch ', best_epoch, '; best_mse:', best_mse, '; best_pearson:',
                      best_pearson, dataset)
            else:
                print(' no improvement since epoch ', best_epoch, '; best_mse, best pearson:', best_mse,
                      best_pearson, dataset)
        draw_loss(train_losses, val_losses, loss_fig_name)
        draw_pearson(val_pearsons, pearson_fig_name)


if __name__ == "__main__":
    batch_size = 128
    parser = argparse.ArgumentParser(description='train model')
    parser.add_argument('--train_batch', type=int, required=False, default=batch_size, help='Batch size training set')
    parser.add_argument('--val_batch', type=int, required=False, default=batch_size, help='Batch size validation set')
    parser.add_argument('--test_batch', type=int, required=False, default=batch_size, help='Batch size test set')
    parser.add_argument('--lr', type=float, required=False, default=0.0001, help='Learning rate')
    parser.add_argument('--num_epoch', type=int, required=False, default=100, help='Number of epoch')
    parser.add_argument('--log_interval', type=int, required=False, default=60, help='Log interval')
    parser.add_argument('--cuda_name', type=str, required=False, default="cuda:0", help='Cuda')
    args = parser.parse_args()
    model = ModDRDSP()
    train_batch = args.train_batch
    val_batch = args.val_batch
    test_batch = args.test_batch
    lr = args.lr
    num_epoch = args.num_epoch
    log_interval = args.log_interval
    cuda_name = args.cuda_name
    device = torch.device(cuda_name if torch.cuda.is_available() else "cpu")
    model.to(device)
    main(model, train_batch, val_batch, test_batch, lr, num_epoch, log_interval, cuda_name)
