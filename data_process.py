import os
import csv
import numpy as np
import pandas as pd
import math
from rdkit import Chem
import networkx as nx
import random
import pickle
import argparse
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import Isomap
from keras_preprocessing import text, sequence
import torch
from torch_geometric import data as DATA
from torch_geometric.data import InMemoryDataset
folder = "data/"
def atom_features(mol, atom):
    hydrogen_donor = Chem.MolFromSmarts("[$([N;!H0;v3,v4&+1]),$([O,S;H1;+0]),n&H1&+0]")
    hydrogen_acceptor = Chem.MolFromSmarts(
        "[$([O,S;H1;v2;!$(*-*=[O,N,P,S])]),$([O,S;H0;v2]),$([O,S;-]),$([N;v3;!$(N-*=[O,N,P,S])]),n&H0&+0,$([o,s;+0;!$([o,s]:n);!$([o,s]:c:n)])]")
    acidic = Chem.MolFromSmarts("[$([C,S](=[O,S,P])-[O;H1,-1])]")
    basic = Chem.MolFromSmarts(
        "[#7;+,$([N;H2&+0][$([C,a]);!$([C,a](=O))]),$([N;H1&+0]([$([C,a]);!$([C,a](=O))])[$([C,a]);!$([C,a](=O))]),$([N;H0&+0]([C;!$(C(=O))])([C;!$(C(=O))])[C;!$(C(=O))])]")
    hydrogen_donor_match = sum(mol.GetSubstructMatches(hydrogen_donor), ())
    hydrogen_acceptor_match = sum(mol.GetSubstructMatches(hydrogen_acceptor), ())
    acidic_match = sum(mol.GetSubstructMatches(acidic), ())
    basic_match = sum(mol.GetSubstructMatches(basic), ())
    ring_info = mol.GetRingInfo()
    atom_idx = atom.GetIdx()
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding(atom.GetTotalDegree(), [0, 1, 2, 3, 4, 5]) +
                    one_of_k_encoding(atom.GetFormalCharge(), [-1, -2, 1, 2, 0]) +
                    one_of_k_encoding_unk(atom.GetChiralTag(), [0, 1, 2, 3]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4]) +
                    one_of_k_encoding_unk(atom.GetHybridization(), [Chem.rdchem.HybridizationType.SP,
                                                                    Chem.rdchem.HybridizationType.SP2,
                                                                    Chem.rdchem.HybridizationType.SP3,
                                                                    Chem.rdchem.HybridizationType.SP3D,
                                                                    Chem.rdchem.HybridizationType.SP3D2]) +
                    one_of_k_encoding_unk(atom.GetExplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8]) +
                    [atom.GetIsAromatic()] +
                    [atom.GetMass() * 0.01] +
                    [atom_idx in hydrogen_acceptor_match] +
                    [atom_idx in hydrogen_donor_match] +
                    [atom_idx in acidic_match] +
                    [atom_idx in basic_match] +
                    [ring_info.IsAtomInRingOfSize(atom_idx, 3),
                     ring_info.IsAtomInRingOfSize(atom_idx, 4),
                     ring_info.IsAtomInRingOfSize(atom_idx, 5),
                     ring_info.IsAtomInRingOfSize(atom_idx, 6),
                     ring_info.IsAtomInRingOfSize(atom_idx, 7),
                     ring_info.IsAtomInRingOfSize(atom_idx, 8)]
                    )

def bond_features(mol, bond):
    """
    Builds a feature vector for a bond.

    :param bond: A RDKit bond.
    :return: A list containing the bond features.
    """
    # 取出化学键上的两个原子并计算两个原子的特征
    atom1 = bond.GetBeginAtom()  # 获取化学键的起始原子
    atom2 = bond.GetEndAtom()  # 获取化学键的结束原子
    atom1_features = atom_features(mol, atom1)
    atom2_features = atom_features(mol, atom2)
    atom_features_all = (atom1_features + atom2_features) / 2
    bond_feature = np.array(one_of_k_encoding_unk(int(bond.GetStereo()), list(range(6)))
                             + [bond.GetIsAromatic()]
                             + [bond.GetIsConjugated()]
                             + [bond.IsInRing()]
                             + one_of_k_encoding_unk(bond.GetBondTypeAsDouble(), [1, 1.5, 2, 3])
                             )
    bond_feature = np.concatenate((atom_features_all, bond_feature))

    return bond_feature

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))

def smile_to_graph(smile):
    mol = Chem.MolFromSmiles(smile)

    c_size = mol.GetNumAtoms()
    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(mol, atom)
        features.append(feature / sum(feature))  # 归一化之后添加

    b_size = mol.GetNumBonds()
    bond_features_list = []
    edges = []
    for bond in mol.GetBonds():
        bond_feature = bond_features(mol, bond)  # 提取化学键特征
        bond_features_list.append(bond_feature / sum(bond_feature))
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])

    g = nx.Graph(edges).to_directed()
    edge_index = []
    for e1, e2 in g.edges:
        edge_index.append([e1, e2])

    return c_size, features, edge_index, b_size, bond_features_list, edges

def save_drug_smile():
    reader = csv.reader(open(folder + "drug_smiles.csv"))
    next(reader, None)

    drug_dict = {}
    drug_smile = []

    for item in reader:
        name = item[0]
        smile = item[2]

        if name in drug_dict:
            pos = drug_dict[name]
        else:
            pos = len(drug_dict)
            drug_dict[name] = pos
        drug_smile.append(smile)

    smile_graph = {}
    for smile in drug_smile:
        g = smile_to_graph(smile)
        smile_graph[smile] = g
    print("save_drug_smile down!")
    return drug_dict, drug_smile, smile_graph

def save_cell_mut_matrix():
    f = open(folder + "PANCANCER_Genetic_feature.csv")
    reader = csv.reader(f)
    next(reader)
    features = {}
    cell_dict = {}
    mut_dict = {}
    matrix_list = []

    for item in reader:
        cell_id = item[1]
        mut = item[5]
        is_mutated = int(item[6])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))

    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    print("save_cell_mut_matrix down!")
    return cell_dict, cell_feature

def save_cell_meth_matrix():
    f = open(folder + "METH_CELLLINES_BEMs_PANCAN2.csv")
    reader = csv.reader(f)
    firstRow = next(reader)
    cell_dict = {}
    matrix_list = []
    mut_dict = {}
    for item in reader:
        cell_id = item[1]
        mut = item[2]
        is_mutated = int(item[3])

        if mut in mut_dict:
            col = mut_dict[mut]
        else:
            col = len(mut_dict)
            mut_dict[mut] = col

        if cell_id in cell_dict:
            row = cell_dict[cell_id]
        else:
            row = len(cell_dict)
            cell_dict[cell_id] = row
        if is_mutated == 1:
            matrix_list.append((row, col))

    cell_feature = np.zeros((len(cell_dict), len(mut_dict)))

    for item in matrix_list:
        cell_feature[item[0], item[1]] = 1

    with open('mut_dict', 'wb') as fp:
        pickle.dump(mut_dict, fp)
    print("save_cell_meth_matrix down!")
    return cell_dict, cell_feature

def save_cell_oge_matrix():
    f = open(folder + "Cell_line_RMA_proc_basalExp.txt")
    line = f.readline()
    elements = line.split()
    cell_names = []
    feature_names = []
    cell_dict = {}
    i = 0
    for cell in range(2, len(elements)):  # 遍历每一列数据，查询所有的细胞名
        if i < 500:
            cell_name = elements[cell].replace("DATA.", "")
            cell_names.append(cell_name)
            cell_dict[cell_name] = []

    min = 0
    max = 12
    for line in f.readlines():
        elements = line.split("\t")
        if len(elements) < 2:
            print(line)
            continue
        feature_names.append(elements[1])

        for i in range(2, len(elements)):
            cell_name = cell_names[i - 2]
            value = float(elements[i])
            if min == 0:
                min = value
            if value < min:
                min = value
            if max < value:
                value = max
            cell_dict[cell_name].append(value)
    cell_feature = []
    for cell_name in cell_names:
        for i in range(0, len(cell_dict[cell_name])):
            cell_dict[cell_name][i] = (cell_dict[cell_name][i] - min) / (max - min)
        cell_dict[cell_name] = np.asarray(cell_dict[cell_name])
        cell_feature.append(np.asarray(cell_dict[cell_name]))

    cell_feature = np.asarray(cell_feature)
    i = 0
    for cell in list(cell_dict.keys()):
        cell_dict[cell] = i
        i += 1
    print("save_cell_oge_matrix down!")
    return cell_dict, cell_feature

def smile_w2v_pad(smile):
    maxlen_ = 150
    victor_size = 100
    smile = [' '.join(s) for s in smile]
    tokenizer = text.Tokenizer(num_words=100, lower=False, filters=" ")  # num_words=100, lower=False, filters="　"
    # 对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示。
    tokenizer.fit_on_texts(smile)
    smile_ = sequence.pad_sequences(tokenizer.texts_to_sequences(smile), maxlen=maxlen_)  # 序列化 0填充
    word_index = tokenizer.word_index
    nb_words = len(word_index)  # 371
    smileVec_model = {}
    with open("data/Atom.vec", encoding='utf8') as f:
        for line in f:  # 遍历每一行79
            values = line.rstrip().rsplit(' ')
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            smileVec_model[word] = coefs
    count = 0
    embedding_matrix = np.zeros((nb_words + 1, victor_size))  # 14,100     (15, 100)
    for word, i in word_index.items():
        embedding_glove_vector = smileVec_model[word] if word in smileVec_model else None
        if embedding_glove_vector is not None:
            count += 1
            embedding_matrix[i] = embedding_glove_vector
        else:
            unk_vec = np.random.random(victor_size) * 0.5
            unk_vec = unk_vec - unk_vec.mean()
            embedding_matrix[i] = unk_vec

    del smileVec_model
    print("save smile_w2v_pad down!")
    return smile_, word_index, embedding_matrix

def save_mix_drug_cell_matrix(choice):
    f = open(folder + "PANCANCER_IC.csv")
    reader = csv.reader(f)
    next(reader)

    cell_dict_mut, cell_feature_mut = save_cell_mut_matrix()
    cell_dict_meth, cell_feature_meth = save_cell_meth_matrix()
    cell_dict_ge, cell_feature_ge = save_cell_oge_matrix()
    drug_dict, drug_smile, smile_graph = save_drug_smile()
    smile_, _, smi_embedding_matrix = smile_w2v_pad(drug_smile)
    if not os.path.exists("data/smi_embedding_matrix.npy"):
        np.save("data/smi_embedding_matrix.npy", smi_embedding_matrix)
    temp_data = []
    bExist = np.zeros((len(drug_dict), len(cell_dict_mut)))

    for item in reader:
        drug = item[0]
        cell = item[3]
        ic50 = item[8]
        ic50 = 1 / (1 + pow(math.exp(float(ic50)), -0.1))
        temp_data.append((drug, cell, ic50))

    xd = []
    xd_seq = []
    xc_mut = []
    xc_meth = []
    xc_ge = []
    y = []
    lst_drug = []
    lst_cell = []
    random.seed(42)
    random.shuffle(temp_data)

    if choice == 0:
        # Kernel PCA
        kpca = KernelPCA(n_components=1000, kernel='rbf', gamma=131, random_state=42)
        cell_feature_ge = kpca.fit_transform(cell_feature_ge)
    elif choice == 1:
        # PCAc
        pca = PCA(n_components=1000)
        cell_feature_ge = pca.fit_transform(cell_feature_ge)
    else:
        # Isomap
        isomap = Isomap(n_components=480)
        cell_feature_ge = isomap.fit_transform(cell_feature_ge)

    for data in temp_data:
        drug, cell, ic50 = data
        if drug in drug_dict and cell in cell_dict_ge and cell in cell_dict_meth:  # and cell in cell_feature_mut
            xd.append(drug_smile[drug_dict[drug]])
            xd_seq.append(smile_[drug_dict[drug]])
            xc_mut.append(cell_feature_mut[cell_dict_mut[cell]])
            xc_ge.append(cell_feature_ge[cell_dict_ge[cell]])
            xc_meth.append(cell_feature_meth[cell_dict_meth[cell]])

            y.append(ic50)
            bExist[drug_dict[drug], cell_dict_mut[cell]] = 1

            lst_drug.append(drug)
            lst_cell.append(cell)

    xd = np.asarray(xd)
    xd_seq = np.asarray(xd_seq)
    xc_mut = np.asarray(xc_mut)
    xc_ge = np.asarray(xc_ge)
    xc_meth = np.asarray(xc_meth)
    y = np.asarray(y)

    pd.DataFrame(np.column_stack((np.array(lst_drug), np.array(lst_cell), y))).to_csv('data/data.csv', index=False)

    size = int(xd.shape[0] * 0.8)
    size1 = int(xd.shape[0] * 0.9)

    xd_train = xd[:size]
    xd_val = xd[size:size1]
    xd_test = xd[size1:]

    xd_seq_train, xd_seq_val, xd_seq_test = xd_seq[:size], xd_seq[size:size1], xd_seq[size1:]

    xc_ge_train = xc_ge[:size]
    xc_ge_val = xc_ge[size:size1]
    xc_ge_test = xc_ge[size1:]

    xc_meth_train = xc_meth[:size]
    xc_meth_val = xc_meth[size:size1]
    xc_meth_test = xc_meth[size1:]

    xc_mut_train = xc_mut[:size]
    xc_mut_val = xc_mut[size:size1]
    xc_mut_test = xc_mut[size1:]

    y_train = y[:size]
    y_val = y[size:size1]
    y_test = y[size1:]
    dataset = 'GDSC'
    print('preparing ', dataset + '_train.pt in pytorch format!')
    train_data = TestbedDataset(root='data',
                                dataset=dataset + '_train_mix',
                                xd=xd_train,
                                xd_seq=xd_seq_train,
                                xt_ge=xc_ge_train,
                                xt_meth=xc_meth_train,
                                xt_mut=xc_mut_train,
                                y=y_train,
                                smile_graph=smile_graph)
    val_data = TestbedDataset(root='data',
                              dataset=dataset + '_val_mix',
                              xd=xd_val,
                              xd_seq=xd_seq_val,
                              xt_ge=xc_ge_val,
                              xt_meth=xc_meth_val,
                              xt_mut=xc_mut_val,
                              y=y_val,
                              smile_graph=smile_graph)
    test_data = TestbedDataset(root='data',
                               dataset=dataset + '_test_mix',
                               xd=xd_test,
                               xd_seq=xd_seq_test,
                               xt_ge=xc_ge_test,
                               xt_meth=xc_meth_test,
                               xt_mut=xc_mut_test,
                               y=y_test,
                               smile_graph=smile_graph)
    print("build data complete")

class TestbedDataset(InMemoryDataset):
    def __init__(self, root='data', dataset='GDSC',
                 xd=None, xd_seq=None,
                 xt_ge=None, xt_meth=None, xt_mut=None, y=None,
                 smile_graph=None, test_drug_dict=None):
        super(TestbedDataset, self).__init__(root)
        self.dataset = dataset
        if os.path.isfile(self.processed_paths[0]):
            print('Pre-processed data found: {}, loading ...'.format(self.processed_paths[0]))
            self.data, self.slices = torch.load(self.processed_paths[0])
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths[0]))
            self.process(xd, xd_seq, xt_ge, xt_meth, xt_mut, y, smile_graph, test_drug_dict)
            self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        pass

    @property
    def processed_file_names(self):
        return [self.dataset + '.pt']
    def _process(self):
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, xd, xd_seq, xt_ge, xt_meth, xt_mut, y, smile_graph, test_drug_dict):
        assert (len(xd) == len(xt_ge) and len(xt_ge) == len(y)) and len(y) == len(xt_meth) and len(xt_meth) == len(
            xt_mut), "The four lists must be the same length!"
        data_list = []  # 118158
        data_len = len(xd)
        print(data_len)
        for i in range(data_len):
            print('Converting SMILES to graph: {}/{}'.format(i + 1, data_len))
            smiles = xd[i]
            target_ge = xt_ge[i]
            target_meth = xt_meth[i]
            target_mut = xt_mut[i]
            labels = y[i]
            smiles_seq = xd_seq[i]
            c_size, features, edge_index, b_size, bond_features_list, edges = smile_graph[smiles]
            GCNData = DATA.Data(x=torch.Tensor(features),
                                xc=torch.Tensor(bond_features_list),
                                edge_index1=torch.LongTensor(edges).transpose(1, 0),
                                edge_index2=torch.LongTensor(edge_index).transpose(1, 0),
                                y=torch.FloatTensor([labels]))
            GCNData.target_ge = torch.FloatTensor([target_ge])
            GCNData.target_meth = torch.FloatTensor([target_meth])
            GCNData.target_mut = torch.FloatTensor([target_mut])
            GCNData.smiles = smiles
            GCNData.smiles_seq = torch.tensor([smiles_seq], dtype=torch.long)

            # Data(x=[36, 78], edge_index=[2, 80], y=[1],
            # target_ge=[1, 1000], target_meth=[1, 377], target_mut=[1, 735],
            # smiles='COC1=C(C=C(C=N1)C2=CC3=C(C=CN=C3C=C2)C4=CN=NC=C4)NS(=O)(=O)C5=C(C=C(C=C5)F)F')

            GCNData.__setitem__('c_size', torch.LongTensor([c_size]))
            GCNData.__setitem__('b_size', torch.LongTensor([b_size]))
            data_list.append(GCNData)
        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        print('Graph construction done. Saving to file.')
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='prepare dataset to train model')
    parser.add_argument('--choice', type=int, required=False, default=0, help='0.KernelPCA, 1.PCA, 2.Isomap')
    args = parser.parse_args()
    choice = args.choice
    save_mix_drug_cell_matrix(choice)