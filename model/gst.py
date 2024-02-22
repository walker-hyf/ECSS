import numpy as np
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class MultiHeadAttention(nn.Module):
    """
    input:
        query --- [N, T_q, query_dim]
        key --- [N, T_k, key_dim]
    output:
        out --- [N, T_q, num_units]
    """

    def __init__(self, query_dim, key_dim, num_units, num_heads):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.key_dim = key_dim

        self.W_query = nn.Linear(in_features=query_dim, out_features=num_units, bias=False)
        self.W_key = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)
        self.W_value = nn.Linear(in_features=key_dim, out_features=num_units, bias=False)

    def forward(self, query, key):
        querys = self.W_query(query)  # [N, T_q, num_units]
        keys = self.W_key(key)  # [N, T_k, num_units]
        values = self.W_value(key)

        split_size = self.num_units // self.num_heads
        querys = torch.stack(torch.split(querys, split_size, dim=2), dim=0)  # [h, N, T_q, num_units/h]
        keys = torch.stack(torch.split(keys, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]
        values = torch.stack(torch.split(values, split_size, dim=2), dim=0)  # [h, N, T_k, num_units/h]

        # score = softmax(QK^T / (d_k ** 0.5))
        scores = torch.matmul(querys, keys.transpose(2, 3)).contiguous()  # [h, N, T_q, T_k]
        scores = scores / (self.key_dim ** 0.5)
        scores = F.softmax(scores, dim=3)

        # out = score * V
        out = torch.matmul(scores, values).contiguous()  # [h, N, T_q, num_units/h]
        out = torch.cat(torch.split(out, 1, dim=0), dim=3).squeeze(0)  # [N, T_q, num_units]

        return out

class ReferenceEncoder(nn.Module):
    """
    inputs --- [N, Ty/r, n_mels*r]  mels
    outputs --- [N, ref_enc_gru_size]
    """

    def __init__(self):
        super().__init__()
        ref_enc_filters = [32, 32, 64, 64, 128, 128]
        n_mel_channels = 80
        ref_enc_gru_size = 128

        K = len(ref_enc_filters)
        filters = [1] + ref_enc_filters

        convs = [nn.Conv2d(in_channels=filters[i],
                           out_channels=filters[i + 1],
                           kernel_size=(3, 3),
                           stride=(2, 2),
                           padding=(1, 1)) for i in range(K)]
        self.convs = nn.ModuleList(convs)
        self.bns = nn.ModuleList(
            [nn.BatchNorm2d(num_features=ref_enc_filters[i])
             for i in range(K)])

        out_channels = self.calculate_channels(n_mel_channels, 3, 2, 1, K)
        self.gru = nn.GRU(input_size=ref_enc_filters[-1] * out_channels,
                          hidden_size=ref_enc_gru_size,
                          batch_first=True)
        self.n_mel_channels = n_mel_channels
        self.ref_enc_gru_size = ref_enc_gru_size

    def forward(self, inputs, input_lengths=None):
        assert inputs.size(-1) == self.n_mel_channels
        out = inputs.unsqueeze(1)
        for conv, bn in zip(self.convs, self.bns):
            out = conv(out)
            out = bn(out)
            out = F.relu(out)

        out = out.transpose(1, 2).contiguous()  # [N, Ty//2^K, 128, n_mels//2^K]
        N, T = out.size(0), out.size(1)
        out = out.contiguous().view(N, T, -1)  # [N, Ty//2^K, 128*n_mels//2^K]

        if input_lengths is not None:
            # print(input_lengths.cpu().numpy(), 2, len(self.convs))
            input_lengths = (input_lengths.cpu().numpy() / 2 ** len(self.convs))
            input_lengths = max(input_lengths.round().astype(int), [1])
            # print(input_lengths, 'input lengths')
            out = nn.utils.rnn.pack_padded_sequence(
                out, input_lengths, batch_first=True, enforce_sorted=False)

        self.gru.flatten_parameters()
        _, out = self.gru(out)
        return out.squeeze(0)

    def calculate_channels(self, l, kernel_size, stride, pad, n_convs):
        for _ in range(n_convs):
            l = (l - kernel_size + 2 * pad) // stride + 1
        return l

# Style Token Layer
class STL(nn.Module):
    """
    inputs --- [N, token_embedding_size//2]
    """

    def __init__(self, token_embedding_size):
        super().__init__()
        token_num = 10
        num_heads = 8
        self.embed = nn.Parameter(torch.FloatTensor(token_num, token_embedding_size // num_heads))
        d_q = token_embedding_size // 2
        d_k = token_embedding_size // num_heads
        self.attention = MultiHeadAttention(
            query_dim=d_q, key_dim=d_k, num_units=token_embedding_size,
            num_heads=num_heads)

        init.normal_(self.embed, mean=0, std=0.5)

    def forward(self, inputs):
        N = inputs.size(0)
        query = inputs.unsqueeze(1)
        keys = torch.tanh(self.embed).unsqueeze(0).expand(N, -1, -1)  # [N, token_num, token_embedding_size//num_heads]
        # print(query.shape, keys.shape)
        style_embed = self.attention(query, keys)

        return style_embed

class GST(nn.Module):
    def __init__(self, token_embedding_size, classes_):
        super().__init__()
        self.encoder = ReferenceEncoder()
        self.stl = STL(token_embedding_size)

        self.categorical_layer = nn.Linear(token_embedding_size, classes_)

    def forward(self, inputs, input_lengths=None):
        enc_out = self.encoder(inputs, input_lengths=input_lengths)
        # print(enc_out.shape)
        style_embed = self.stl(enc_out)

        cat_prob = F.softmax(self.categorical_layer(style_embed.squeeze(0)), dim=-1)
        # print(style_embed.shape, cat_prob.shape)
        return (style_embed.squeeze(0), cat_prob)


# load_npy = np.load("E:\\dialog_TTS\\DailyTalk_interspeech23\\preprocessed_data\\DailyTalk\\mel_frame\\1-mel-5_1_d2136.npy")
# mel = torch.tensor(load_npy).unsqueeze(0)
# # print(load_npy)
# gst = GST(256,7)
# out = gst(mel)
# print(out[0].shape)
# print(out[1])