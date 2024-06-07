import torch
from torch import nn
from torch.nn.utils import weight_norm
import torch.nn.functional as F
import math


def get_lap(adj):
    adj = torch.nan_to_num(adj / torch.sum(adj, dim=-1).unsqueeze(dim=-1), nan=0)
    return adj


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super(FeedForward, self).__init__()
        self._linear1 = nn.Linear(d_model, d_ff)
        self._linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        return self._linear2(F.relu(self._linear1(x)))


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # 创建位置编码矩阵
        pe = torch.zeros(self.max_seq_len, self.d_model).cuda()

        # 计算位置编码的值
        position = torch.arange(0, self.max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.d_model, 2) * (-math.log(10000.0) / self.d_model))

        # 调整位置编码矩阵的值
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        # 添加一个维度作为可学习的参数
        pe = pe.unsqueeze(0)  # (1,6,h)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # 将位置编码加到输入张量中
        # (32,115,6,h)
        x = x + self.pe[:, :x.shape[2]]
        return x


class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        """
        其实这就是一个裁剪的模块，裁剪多出来的padding
        """
        return x[:, :, :-self.chomp_size].contiguous()


class TCNBlock(nn.Module):
    def __init__(self, hidden_size, padding, dilation, kernel_size, dropout):
        super().__init__()

        self.conv1 = weight_norm(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(kernel_size, 1),
                      padding=(padding, 0),
                      stride=1,
                      dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(
            nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(kernel_size, 1),
                      padding=(padding, 0),
                      stride=1,
                      dilation=dilation))
        self.chomp2 = Chomp1d(padding)  # 裁剪掉多出来的padding部分，维持输出时间步为seq_len
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        # self.residual =  nn.Conv2d(in_channels=hidden_size, out_channels=hidden_size, kernel_size=(1, 1))

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.init_weights()

    def init_weights(self):
        """
        参数初始化
        """
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        # self.residual.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # x:(64,T,114,32)
        x = x.transpose(2, 3).transpose(1, 2)  # (64,32,6,114)
        out = self.net(x)  # (64,32,6,114)
        out = out + x
        out = out.transpose(1, 2).transpose(2, 3)
        return F.relu(out)


class TCN(nn.Module):
    def __init__(self, hidden_size, paddings, dilations, kernel_size, dropout):
        super().__init__()
        layers = []
        for i in range(len(dilations)):
            dilation_size = dilations[i]  # 膨胀系数：1，2，2
            padding_size = paddings[i]
            layers += [
                TCNBlock(hidden_size=hidden_size, dilation=dilation_size, padding=padding_size, kernel_size=kernel_size,
                         dropout=dropout)]
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        # x:(64,T,114,32)
        return self.network(x)


class Matmul(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, x, A):
        x = torch.matmul(A, x)
        return x.contiguous()


class DiffusionConv(nn.Module):
    def __init__(self, c_in, c_out, dropout, support_len=2, order=2):
        super().__init__()
        self.nconv = Matmul()
        c_in = (order * support_len + 1) * c_in
        self.mlp = nn.Linear(c_in, c_out)
        self.dropout = dropout
        self.order = order

    def forward(self, x, dist_adj, heading_adj):
        support = [dist_adj, heading_adj]
        out = [x]
        for a in support:
            x1 = self.nconv(x, a)
            out.append(x1)
            for k in range(2, self.order + 1):
                x2 = self.nconv(x1, a)
                out.append(x2)
                x1 = x2

        h = torch.cat(out, dim=-1)
        h = self.mlp(h)
        h = F.dropout(h, self.dropout, training=self.training)
        return h


class GlobalSpatialMHA(nn.Module):
    def __init__(self, heads, hidden_size):
        super().__init__()
        self.d_v = hidden_size // heads
        self.heads = heads
        self.hidden_size = hidden_size
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_q, input_k, input_v):
        # x(64,115,h)
        batch, num_object = input_v.shape[0], input_v.shape[1]
        Q = self.W_Q(input_q).reshape(batch, num_object, self.heads, self.d_v).transpose(1, 2)  # (64, heads, 115, d_v)
        K = self.W_K(input_k).reshape(batch, num_object, self.heads, self.d_v).transpose(1, 2)  # (64, heads, 115, d_v)
        V = self.W_V(input_v).reshape(batch, num_object, self.heads, self.d_v).transpose(1, 2)  # (64, heads, 115, d_v)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_v ** 0.5)  # (64, heads, 115,115)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)  # (64, heads, 115,h)
        context = context.transpose(1, 2)
        context = context.reshape(batch, num_object, self.heads * self.d_v)  # (64, 115, heads*d_v)
        context = self.fc(context)  # x:(64,115,h)
        return context


class SpatialMultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_size, history_frames):
        super().__init__()
        self.d_v = hidden_size // heads
        self.heads = heads
        self.history_frames = history_frames
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)

    def forward(self, input_q, input_k, input_v):
        # x:(64,6,115,h)
        batch, num_object = input_q.shape[0], input_q.shape[2]
        Q = self.W_Q(input_q).reshape(batch, self.history_frames, num_object, self.heads, self.d_v).transpose(2, 3)
        K = self.W_K(input_k).reshape(batch, self.history_frames, num_object, self.heads, self.d_v).transpose(2, 3)
        V = self.W_V(input_v).reshape(batch, self.history_frames, num_object, self.heads, self.d_v).transpose(2, 3)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_v ** 0.5)  # (64, 6, heads, 115,115)
        attention = F.softmax(attention, dim=-1)
        context = torch.matmul(attention, V)  # (64, 6, heads, 115,h)
        context = context.transpose(2, 3)
        context = context.reshape(batch, self.history_frames, num_object,
                                  self.heads * self.d_v)  # (64, 6, 115, heads*h)
        context = self.fc(context)  # x:(64,6,115,h)
        # context = self.dropout(context)  # x:(64,115,6,h)
        return context


class TemporalMultiHeadAttention(nn.Module):
    def __init__(self, heads, hidden_size, history_frames, dropout):
        super().__init__()
        self.d_v = hidden_size // heads
        self.heads = heads
        self.history_frames = history_frames
        self.hidden_size = hidden_size
        self.W_Q = nn.Linear(hidden_size, hidden_size)
        self.W_K = nn.Linear(hidden_size, hidden_size)
        self.W_V = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size, hidden_size)
        self.masks = torch.tril(torch.ones((history_frames, history_frames)), diagonal=0).cuda()
        self.layerNorm1 = nn.LayerNorm(hidden_size)
        self.layerNorm2 = nn.LayerNorm(hidden_size)
        self.feedForward = FeedForward(hidden_size, hidden_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, input_q, input_k, input_v):
        residual = input_q
        batch, num_object = input_q.shape[0], input_q.shape[1]
        Q = self.W_Q(input_q).reshape(batch, num_object, self.history_frames, self.heads, self.d_v).transpose(2, 3)
        K = self.W_K(input_k).reshape(batch, num_object, self.history_frames, self.heads, self.d_v).transpose(2, 3)
        V = self.W_V(input_v).reshape(batch, num_object, self.history_frames, self.heads, self.d_v).transpose(2, 3)
        attention = torch.matmul(Q, K.transpose(-1, -2)) / (self.d_v ** 0.5)  # (64, 115, heads, 6, 6)
        context = torch.matmul(F.softmax(attention, dim=-1), V)  # (64, 115, heads, 6,h)
        context = context.transpose(2, 3)
        context = context.reshape(batch, num_object, self.history_frames,
                                  self.heads * self.d_v)  # (64, 115, 6, heads*h)
        context = self.fc(context)  # x:(64,115,6,h)
        context = self.dropout(context)  # x:(64,115,6,h)
        context = self.layerNorm1(context + residual)  # (64,6,114,h)

        context_1 = self.feedForward(context)  # (64,6,114,h)
        context_1 = self.dropout(context_1)  # (64,6,114,h)
        last_out = self.layerNorm2(context_1 + context)  # (64,6,114,h)
        last_out = last_out.transpose(1, 2)
        return last_out


class GlobalSpatial(nn.Module):
    def __init__(self, heads, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.gru = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.gsa = GlobalSpatialMHA(heads, hidden_size)

    def forward(self, x):
        # x(64,T,114,h)
        batch, step, num_object = x.shape[0], x.shape[1], x.shape[2]
        x = x.transpose(1, 2)
        x = x.reshape(batch * num_object, step, self.hidden_size)  # x(64*115,T,h)
        x, gt = self.gru(x)[1][0], self.gru(x)[0]  # (1,64*115,h)
        x = x.squeeze().reshape(batch, num_object, self.hidden_size)  # (64,115,h)
        x = self.gsa(x, x, x)  # (64,115,h)
        x = torch.unsqueeze(x, dim=1)  # (64,1,115,h)
        gt = gt.reshape(batch, num_object, step, self.hidden_size).transpose(1, 2)
        return x, gt  # (64,1,115,h)


class TemporalBlock(nn.Module):
    def __init__(self, heads, history_frames, hidden_size, paddings, dilations, kernel_size, dropout):
        super().__init__()
        self.gt = TemporalMultiHeadAttention(heads, hidden_size, history_frames, dropout)
        # self.pe = PositionalEncoding(hidden_size, 32)
        # self.fc_1 = nn.Linear(hidden_size, hidden_size)
        # self.fc_2 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x):
        x = x.transpose(1, 2)  # (32,115,6,h)
        # x = self.pe(x)

        Q, K, V = x, x, x
        tmha = self.gt(Q, K, V)

        # z = torch.sigmoid(self.fc_1(tmha) + self.fc_2(gt))
        # last_out = z * tmha + (1 - z) * gt
        return tmha


class SpatialBlock(nn.Module):
    def __init__(self, heads, hidden_size, history_frames, dropout):
        super().__init__()
        self.satt = SpatialMultiHeadAttention(heads, hidden_size, history_frames)
        self.dc = DiffusionConv(hidden_size, hidden_size, dropout)
        self.fc_1 = nn.Linear(hidden_size, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, hidden_size)
        self.fc_3 = nn.Linear(hidden_size, hidden_size)
        self.fc_4 = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, dist_adj, heading_adj, gs_out):
        # (64,6,3,115,115)
        Q, K, V = x, x, x
        satt_out = self.satt(Q, K, V)
        gcn_out = self.dc(x, dist_adj, heading_adj)

        z = torch.sigmoid(self.fc_1(gcn_out) + self.fc_2(satt_out))
        last_out = z * gcn_out + (1 - z) * satt_out

        a = torch.sigmoid(self.fc_3(last_out) + self.fc_4(gs_out))
        last_out = a * last_out + (1 - a) * gs_out
        return last_out


class SpatialTemporal(nn.Module):
    def __init__(self, hidden_size, history_frames, heads, paddings, dilations, kernel_size, dropout):
        super().__init__()
        self.gs = GlobalSpatial(heads, hidden_size)
        self.spatial = SpatialBlock(heads, hidden_size, history_frames, dropout)
        self.temporal = TemporalBlock(heads, history_frames, hidden_size, paddings, dilations, kernel_size, dropout)

    def forward(self, x, dist_adj, heading_adj):
        # x:(64,T,114,h), (64,6,115,115), (64,6,115,115)
        gs_out, gt = self.gs(x)
        x = self.spatial(x, dist_adj, heading_adj, gs_out)
        x = self.temporal(x)
        x = x + gt
        return x


# class Seq2Seq(nn.Module):
#     def __init__(self, hidden_size, out_channels, history_frames, future_frames, max_object_num):
#         super().__init__()
#         self.out_channels = out_channels
#         self.hidden_size = hidden_size
#         self.future_frames = future_frames
#         self.max_object_num = max_object_num
#         self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
#         self.gru_cell = nn.GRU(hidden_size + hidden_size, hidden_size, batch_first=True)
#         self.W_e = nn.Linear(hidden_size,hidden_size)
#         self.W_d = nn.Linear(hidden_size,hidden_size)
#         self.W_a = nn.Linear(hidden_size, 1)
#         self.dropout = nn.Dropout(p=0.2)
#         self.fc_2 = nn.Linear(hidden_size, out_channels)
# #         self.reset_parameters()

# #     def reset_parameters(self):
# #         nn.init.xavier_uniform_(self.W_e)
# #         nn.init.xavier_uniform_(self.W_d)
# #         nn.init.xavier_uniform_(self.W_a)

#     def forward(self, h, last_position, teacher_location=None):
#         # (64,6,114,h), (64,114,2)
#         if teacher_location is not None:
#             teacher_location = teacher_location.transpose(1, 2)
#             teacher_location = teacher_location.reshape(-1, teacher_location.shape[2], 2)  # (64*114,6,2)
#         h = h.transpose(1, 2)
#         h = h.reshape(-1, h.shape[2], self.hidden_size)  # (64*114, 6, h)
#         last_position = last_position.reshape(-1, 2).unsqueeze(dim=1)  # (64*114, 1, 2)
#         last_out = torch.zeros((h.shape[0], self.future_frames, 2)).cuda()  # (64*114,6,2)

#         x = torch.zeros((h.shape[0], 1, self.hidden_size)).cuda()  # (64*114,1,h)
#         x = torch.cat([last_position, x], dim=-1)  # (64*114,1,h+2)

#         output, h_t = self.gru(h)  # (64*114,6,h), (1,64*114,h)
#         for step in range(self.future_frames):
#             if step == 0:
#                 new_out, h_t = self.gru_cell(x, h_t)  # (64*114,1,h), (1,64*114,h)
#                 # new_out =   # (64*114, 1, 2)
#                 # new_out = new_out + last_position  # (64*114, 1, 2)
#                 last_out[:, step:step + 1, :] = self.fc_2(new_out)
#             else:
#                 a = F.softmax(self.W_a(torch.tanh(self.W_e(output) + self.W_d(h_t.transpose(0, 1)))),dim=1)  # (64*114,6,1)
#                 c = torch.matmul(output.transpose(1, 2), a).transpose(1, 2)  # (64*114,1,h)
#                 teacher_force = np.random.random() < 0.5
#                 new_out = (teacher_location[:, step - 1:step] if (type(teacher_location) is not type(
#                     None)) and teacher_force else new_out)
#                 new_out, h_t = self.gru_cell(torch.cat([new_out, c], dim=-1), h_t)  # (64*114,1,h), (64*114,1,h)
#                 last_out[:, step:step + 1, :] = self.fc_2(new_out)
#         last_out = last_out.reshape(-1, self.max_object_num, self.future_frames, self.out_channels)
#         last_out = last_out.transpose(1, 2)
#         return last_out


# class Seq2Seq(nn.Module):
#     def __init__(self, hidden_size, out_channels, history_frames, future_frames, max_object_num):
#         super().__init__()
#         self.out_channels = out_channels
#         self.hidden_size = hidden_size
#         self.future_frames = future_frames
#         self.max_object_num = max_object_num
#         self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
#         self.gru_cell = nn.GRU(hidden_size + hidden_size, hidden_size, batch_first=True)
#         self.fc = nn.Linear(out_channels,hidden_size)
#         self.W_e = nn.Linear(hidden_size,hidden_size)
#         self.W_d = nn.Linear(hidden_size,hidden_size)
#         self.W_a = nn.Linear(hidden_size, 1)
#         self.dropout = nn.Dropout(p=0.2)
#         self.fc_2 = nn.Linear(hidden_size, out_channels)

#     def forward(self, h, last_position, teacher_location=None):
#         # (64,6,114,h), (64,114,2)
#         if teacher_location is not None:
#             teacher_location = teacher_location.transpose(1, 2)
#             teacher_location = teacher_location.reshape(-1, teacher_location.shape[2], 2)  # (64*114,6,2)
#             teacher_location = self.fc(teacher_location)
#         h = h.transpose(1, 2)
#         h = h.reshape(-1, h.shape[2], self.hidden_size)  # (64*114, 6, h)
#         last_position = last_position.reshape(-1, 2).unsqueeze(dim=1)  # (64*114, 1, 2)
#         last_position = self.fc(last_position)
#         last_out = torch.zeros((h.shape[0], self.future_frames, 2)).cuda()  # (64*114,6,2)

#         x = torch.zeros((h.shape[0], 1, self.hidden_size)).cuda()  # (64*114,1,h)
#         x = torch.cat([last_position, x], dim=-1)  # (64*114,1,h+2)

#         output, h_t = self.gru(h)  # (64*114,6,h), (1,64*114,h)
#         for step in range(self.future_frames):
#             if step == 0:
#                 new_out, h_t = self.gru_cell(x, h_t)  # (64*114,1,h), (1,64*114,h)
#                 last_out[:, step:step + 1, :] = self.fc_2(new_out)
#             else:
#                 a = F.softmax(self.W_a(torch.tanh(self.W_e(output) + self.W_d(h_t.transpose(0, 1)))),dim=1)  # (64*114,6,1)
#                 c = torch.matmul(output.transpose(1, 2), a).transpose(1, 2)  # (64*114,1,h)
#                 teacher_force = np.random.random() < 0.5
#                 new_out = (teacher_location[:, step - 1:step] if (type(teacher_location) is not type(
#                     None)) and teacher_force else new_out)
#                 new_out, h_t = self.gru_cell(torch.cat([new_out, c], dim=-1), h_t)  # (64*114,1,h), (64*114,1,h)
#                 last_out[:, step:step + 1, :] = self.fc_2(new_out)
#         last_out = last_out.reshape(-1, self.max_object_num, self.future_frames, self.out_channels)
#         last_out = last_out.transpose(1, 2)
#         return last_out


class Model(nn.Module):
    def __init__(self, in_channels, out_channels, heads, hidden_size, layers, history_frames, max_object_num, paddings,
                 dilations, kernel_size, dropout):
        super().__init__()
        self.embed = nn.Linear(in_channels, hidden_size)
        self.layers = layers
        self.hidden_size = hidden_size
        self.history_frames = history_frames
        self.max_object_num = max_object_num
        self.st_block = SpatialTemporal(hidden_size, history_frames, heads, paddings, dilations, kernel_size, dropout)
        self.tcn_1 = TCN(hidden_size, paddings[0], dilations[0], kernel_size[0], dropout)
        self.tcn_2 = TCN(hidden_size, paddings[1], dilations[1], kernel_size[1], dropout)
        self.fc_1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc_2 = nn.Linear(hidden_size, out_channels)
        # self.seq2seq = Seq2Seq(hidden_size,out_channels,history_frames,history_frames,max_object_num)

    def forward(self, x, dist_adj, heading_adj):
        # last_position = x[:,-1,:,:2]
        x = self.embed(x)  # (32,6,115,h)
        residual = x  # (64,T,115,h)

        dist_adj = get_lap(dist_adj)
        heading_adj = get_lap(heading_adj)

        for layer in range(self.layers):
            x = self.st_block(x, dist_adj, heading_adj) + residual  # (64,6,115,h)
            residual = x
        # last_out = self.seq2seq(x,last_position,teacher_location)
        # last_out = self.tcn_1(x)
        last_out = torch.cat([self.tcn_1(x), self.tcn_2(x)], dim=-1)
        last_out = self.fc_1(last_out)
        last_out = self.fc_2(last_out)
        return last_out
