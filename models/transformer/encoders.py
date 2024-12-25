from torch.nn import functional as F
from torch.nn.modules.activation import LeakyReLU
from models.transformer.utils import PositionWiseFeedForward
import torch
from torch import nn
from models.transformer.attention import MultiHeadAttention

class SIE(nn.Module):
    def __init__(self, num_semantics=5, dim=512):
        super().__init__()
        self.num_semantics = num_semantics
        self.dim = dim
        self.weight = nn.Linear(dim, num_semantics)
        self.init_weights()

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, grids):
        bs, grid_num = grids.shape[0], grids.shape[1]
        grids = F.normalize(grids, p=2, dim=-1)
        # (bs, num_semantics, 49)
        weight = F.softmax(self.weight(grids), dim=1).permute(0, 2, 1)
        # (bs, num_semantics, dim)
        semantics = torch.matmul(weight, grids)
        semantics = F.normalize(semantics, p=2, dim=2)
        # [bs, num_semantics * dim]
        semantics = semantics.view(bs, -1)
        semantics = F.normalize(semantics, p=2, dim=1)
        # [bs, num_semantics, dim]
        semantics = semantics.view(bs, self.num_semantics, self.dim)
        return semantics

class MLSA(nn.Module):
    def __init__(self, num_semantics, d_model=512):
        super(MLSA, self).__init__()
        self.num_semantics = num_semantics
        self.SR = nn.LSTM(input_size=d_model, hidden_size=d_model, num_layers=1,
                            bias=True, batch_first=True, bidirectional=False)
        self.SIE = SIE(num_semantics=self.num_semantics, dim=512)
        self.Adaptive_weight = nn.Linear(512 * 2, 1)

    def forward(self, x, layers, cross_att_layer, attention_mask=None, attention_weights=None,
                relative_geometry_weights=None):
        out = x
        semantics = []
        b_s = x.shape[0]
        num_semantics = self.num_semantics + 1  # + global_semantic
        d_model = x.shape[2]

        # SIE
        for l in layers:
            out = l(out, out, out,
                    attention_mask=attention_mask, attention_weights=attention_weights,
                    relative_geometry_weights=relative_geometry_weights)
            semantic = self.SIE(out)
            global_semantic = torch.mean(out, dim=1).unsqueeze(1)  # [bs, 1, 512]
            semantic = torch.cat([semantic, global_semantic], dim=1)  # [bs, num_semantics + 1, 512]
            semantics.append(semantic.reshape(-1, 1, d_model))  # [bs * (num_semantics + 1), 1, 512]

        # SR
        out_lstm = torch.randn(b_s * num_semantics, 1, d_model)
        for i in semantics:
            out_lstm, _ = self.SR(i)  # [bs * (num_semantics + 1), 1, hidden_size]
        refined_semantic = out_lstm.reshape(b_s, num_semantics, d_model)
        # AA
        semantic_out = cross_att_layer(out, refined_semantic, refined_semantic)
        f = torch.cat((out, semantic_out), dim=-1)
        semantic_weight = torch.sigmoid(self.Adaptive_weight(f))
        out = out + semantic_weight * semantic_out
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1, identity_map_reordering=False,
                 attention_module=None, attention_module_kwargs=None):
        super(EncoderLayer, self).__init__()
        self.identity_map_reordering = identity_map_reordering
        self.mhatt = MultiHeadAttention(d_model, d_k, d_v, h, dropout, identity_map_reordering=identity_map_reordering,
                                        attention_module=attention_module,
                                        attention_module_kwargs=attention_module_kwargs)
        self.dropout = nn.Dropout(dropout)
        self.lnorm = nn.LayerNorm(d_model)
        self.pwff = PositionWiseFeedForward(d_model, d_ff, dropout, identity_map_reordering=identity_map_reordering)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None,
                relative_geometry_weights=None):

        att = self.mhatt(queries, keys, values,
                         attention_mask=attention_mask, attention_weights=attention_weights,
                         relative_geometry_weights=relative_geometry_weights)
        att = self.lnorm(queries + self.dropout(att))
        ff = self.pwff(att)
        return ff


class MultiLevelEncoder(nn.Module):
    def __init__(self, N, padding_idx, num_semantics, d_model=512, d_k=64, d_v=64, h=8, d_ff=2048, dropout=.1,
                 identity_map_reordering=False, attention_module=None, attention_module_kwargs=None):
        super(MultiLevelEncoder, self).__init__()
        self.d_model = d_model
        self.dropout = dropout
        self.num_semantics = num_semantics
        self.layers = nn.ModuleList([EncoderLayer(d_model, d_k, d_v, h, d_ff, dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module=attention_module,  # VCGA
                                                  attention_module_kwargs=attention_module_kwargs)  # {'m': args.m}
                                     for _ in range(N)])
        self.Cross_Att_layer = EncoderLayer(d_model, d_k, d_v, h, d_ff,dropout,
                                                  identity_map_reordering=identity_map_reordering,
                                                  attention_module_kwargs=attention_module_kwargs)
        self.MLSA = MLSA(self.num_semantics, d_model)
        self.padding_idx = padding_idx

    def forward(self, input, attention_weights=None, relative_geometry_weights=None):
        # input: (b_s, 49, d_model)
        # (b_s, 1, 1, 49)
        attention_mask = (torch.sum(input, -1) == self.padding_idx).unsqueeze(1).unsqueeze(1)  # (b_s, 1, 1, 50)
        out = self.MLSA(input, self.layers, self.Cross_Att_layer,
                        attention_mask=attention_mask, attention_weights=attention_weights,
                        relative_geometry_weights=relative_geometry_weights)
        return out, attention_mask


class TransformerEncoder(MultiLevelEncoder):
    def __init__(self, N, padding_idx, num_semantics, d_in=2048, **kwargs):
        super(TransformerEncoder, self).__init__(N, padding_idx, num_semantics, **kwargs)
        self.num_semantics = num_semantics
        self.fc = nn.Linear(d_in, self.d_model)
        self.dropout = nn.Dropout(p=self.dropout)
        self.layer_norm = nn.LayerNorm(self.d_model)

    def forward(self, input, attention_weights=None, relative_geometry_weights=None):
        mask = (torch.sum(input, dim=-1) == 0).unsqueeze(-1)
        out = F.relu(self.fc(input))
        out = self.dropout(out)
        out = self.layer_norm(out)
        out = out.masked_fill(mask, 0)
        return super(TransformerEncoder, self).forward(out, attention_weights=attention_weights,
                                                       relative_geometry_weights=relative_geometry_weights)
