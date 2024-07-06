import torch
from torch import nn
import torch.nn.functional as F


class smile2Vec(nn.Module):

    def __init__(self, smi_embedding_matrix, input_size=100, hidden_size=128, dropout=0.2):
        super(smile2Vec, self).__init__()
        self.embed_smile = nn.Embedding(100, 100)
        self.smi_embedding_matrix = smi_embedding_matrix
        self.embed_smile.weight = nn.Parameter(torch.tensor(smi_embedding_matrix, dtype=torch.float32))
        self.embed_smile.weight.requires_grad = True
        self.W_rnn1 = nn.GRU(bidirectional=True, num_layers=2, input_size=input_size, hidden_size=hidden_size)
        self.W_rnn2 = nn.GRU(bidirectional=True, num_layers=2, input_size=2 * hidden_size, hidden_size=hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.uttenc = UttEncoder(hidden_size*2, hidden_size, 'higru-sf')#256

    def forward(self, smiles):
        smile_vectors = self.embed_smile(smiles)  # 获取分子嵌入
        smile_vectors = self.dropout(smile_vectors)  # Dropout层添加到输入 #Size([128, 150, 100])
        after_smile_vectors1, _ = self.W_rnn1(smile_vectors)  # 获取第一层GRU的输出和隐藏状态 Size([128, 150, 256])
        after_smile_vectors1 = torch.relu(after_smile_vectors1)
        s_embed = self.uttenc(after_smile_vectors1)  # torch.Size([16, 300])
        after_smile_vectors2, _ = self.W_rnn2(s_embed)  # 获取第二层GRU的输出和隐藏状态  #Size([128, 150, 256])
        after_smile_vectors2 = torch.relu(after_smile_vectors2)
        after_smile_vectors2 = self.dropout(after_smile_vectors2)
        return torch.mean(after_smile_vectors2, 2)# s_embed#


# Utterance encoder with three types: higru, higru-f, and higru-sf
class UttEncoder(nn.Module):
    def __init__(self, d_word_vec, d_h1, type):
        super(UttEncoder, self).__init__()
        # self.encoder = GRUencoder(d_word_vec, d_h1, num_layers=1)
        # self.d_input = 2 * d_h1
        self.model = type
        if self.model == 'higru-f':
            self.d_input = 2 * d_h1 + d_word_vec
        if self.model == 'higru-sf':
            self.d_input = 4 * d_h1 + d_word_vec
        self.output1 = nn.Sequential(
            nn.Linear(self.d_input, d_h1*2),
            nn.Tanh()
        )

    def forward(self, sents, sa_mask=None):
        """
        :param sents: batch x seq_len x 2*d_h1
        :param lengths: numpy array 1 x batch
        :return: batch x d_h1
        """
        # w_context = self.encoder(sents, lengths)
        w_context = sents
        combined = sents

        if self.model == 'higru-f':
            w_lcont, w_rcont = w_context.chunk(2, -1)
            combined = [w_lcont, sents, w_rcont]
            combined = torch.cat(combined, dim=-1)
        if self.model == 'higru-sf':
            w_lcont, w_rcont = w_context.chunk(2, -1)
            sa_lcont, _ = get_attention(w_lcont, w_lcont, w_lcont, attn_mask=sa_mask)
            sa_rcont, _ = get_attention(w_rcont, w_rcont, w_rcont, attn_mask=sa_mask)
            combined = [sa_lcont, w_lcont, sents, w_rcont, sa_rcont]
            combined = torch.cat(combined, dim=-1)

        output1 = self.output1(combined)#torch.Size([128, 150, 256])
        # output = torch.max(output1, dim=1)[0]

        return output1


# Dot-product attention
def get_attention(q, k, v, attn_mask=None):
    """
    :param : (batch, seq_len, seq_len)
    :return: (batch, seq_len, seq_len)
    """
    attn = torch.matmul(q, k.transpose(1, 2))
    if attn_mask is not None:
        attn.data.masked_fill_(attn_mask, -1e10)

    attn = F.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)
    return output, attn

#
# class smile2Vec(nn.Module):
#     def __init__(self, smi_embedding_matrix, input_size=100, hidden_size=128, dropout=0.2):
#         super(smile2Vec, self).__init__()
#         self.embed_smile = nn.Embedding(100, 100)
#         self.smi_embedding_matrix = smi_embedding_matrix
#         self.embed_smile.weight = nn.Parameter(torch.tensor(smi_embedding_matrix, dtype=torch.float32))
#         self.embed_smile.weight.requires_grad = True
#         self.num_groups = 2  # 分组数量
#         self.group_rnn1 = nn.ModuleList(
#             [nn.GRU(bidirectional=True, num_layers=2, input_size=input_size // self.num_groups, hidden_size=hidden_size)
#              for _ in range(self.num_groups)])
#         self.group_rnn2 = nn.ModuleList(
#             [nn.GRU(bidirectional=True, num_layers=2, input_size=2 * hidden_size, hidden_size=hidden_size) for _ in
#              range(self.num_groups)])
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, smiles):
#         smile_vectors = self.embed_smile(smiles)  # 获取分子嵌入
#         smile_vectors = self.dropout(smile_vectors)  # Dropout层添加到输入
#
#         group_size = smile_vectors.size(2) // self.num_groups
#         group_outputs1 = []
#         group_outputs2 = []
#
#         for i in range(self.num_groups):
#             group_input = smile_vectors[:, :, i * group_size:(i + 1) * group_size]
#             group_output1, _ = self.group_rnn1[i](group_input)
#             group_output1 = torch.relu(group_output1)
#             group_outputs1.append(group_output1)
#
#             group_output2, _ = self.group_rnn2[i](group_output1)
#             group_output2 = torch.relu(group_output2)
#             group_outputs2.append(group_output2)
#
#         after_smile_vectors2 = torch.cat(group_outputs2, dim=2)
#         after_smile_vectors2 = self.dropout(after_smile_vectors2)
#         return torch.mean(after_smile_vectors2, dim=2)


# class smile2Vec(nn.Module):
#
#     def __init__(self, smi_embedding_matrix, input_size=100, hidden_size=128, dropout=0.2):
#         super(smile2Vec, self).__init__()
#         self.embed_smile = nn.Embedding(100, 100)
#         self.smi_embedding_matrix = smi_embedding_matrix
#         self.embed_smile.weight = nn.Parameter(torch.tensor(smi_embedding_matrix, dtype=torch.float32))
#         self.embed_smile.weight.requires_grad = True
#         self.W_rnn1 = nn.GRU(bidirectional=True, num_layers=2, input_size=input_size, hidden_size=hidden_size)
#         self.W_rnn2 = nn.GRU(bidirectional=True, num_layers=2, input_size=2 * hidden_size, hidden_size=hidden_size)
#         self.dropout = nn.Dropout(dropout)
#
#     def forward(self, smiles):
#         smile_vectors = self.embed_smile(smiles)  # 获取分子嵌入
#         smile_vectors = self.dropout(smile_vectors)  # Dropout层添加到输入 #Size([128, 150, 100])
#         after_smile_vectors1, _ = self.W_rnn1(smile_vectors)  # 获取第一层GRU的输出和隐藏状态 Size([128, 150, 256])
#         after_smile_vectors1 = torch.relu(after_smile_vectors1)
#         after_smile_vectors2, _ = self.W_rnn2(after_smile_vectors1)  # 获取第二层GRU的输出和隐藏状态  #Size([128, 150, 256])
#         after_smile_vectors2 = torch.relu(after_smile_vectors2)
#         after_smile_vectors2 = self.dropout(after_smile_vectors2)
#         return torch.mean(after_smile_vectors2, 2)
