# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Various positional encodings for the transformer.
"""
import math
import torch
from torch import nn


class PositionEmbeddingSine(nn.Module):
    """
    This is a more standard version of the position embedding, very similar to the one
    used by the Attention is all you need paper, generalized to work on images.
    """
    # 256, normalize=True
    def __init__(self, num_pos_feats=256, temperature=10000, normalize=False, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        if scale is None:
            scale = 2 * math.pi
        self.scale = scale

    # (b_s, 7, 7, feature_dim)
    def forward(self, x, mask=None):
        if mask is None:
            # (b_s, 7, 7) 全0
            mask = torch.zeros(x.shape[:-1], dtype=torch.bool, device=x.device)
        # 全1
        not_mask = (mask == False)
        """
        torch.cumsum(input, dim, *, dtype=None, out=None) → Tensor
        第dim维的第一个元素不变，其余元素累加
        """
        # not_mask[i;0;]不变，其余not_mask[i;1-6;]依次累加
        # not_mask[i;0;] 全为1，not_mask[i;1;]全2，not_mask[i;2;]全3,...
        # (b_s,7,7)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        # not_mask[i;j;0]不变，其余not_mask[i;j;1-6]依次累加
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            # 1-7 -> 7-1
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        # (self.num_pos_feats = 256,)
        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        # 0 <= (2 * (dim_t // 2)) / self.num_pos_feats < 1
        # dim_t[i] < self.num_pos_feats
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        # x_embed: (b_s, 7, 7) -> (b_s, 7, 7, self.num_pos_feats)
        # dim_t: (self.num_pos_feats ,) -> (b_s, 7, 7, self.num_pos_feats)
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        """
        stack(): 沿着一个新维度对输入张量序列进行连接。 序列中所有的张量都应该为相同形状。
        """
        # (b_s, 7, 7, 1) -> (b_s, 7, 7, self.num_pos_feats/2, 2) -> (b_s, 7, 7, self.num_pos_feats)
        # 从第3维开始展平
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        # (b_s, 7, 7, self.num_pos_feats) -> (b_s, 7, 7, self.num_pos_feats * 2)
        pos = torch.cat((pos_y, pos_x), dim=3)  # .permute(0, 3, 1, 2)
        # (b_s, 7 * 7, self.num_pos_feats * 2)
        pos = pos.flatten(1, 2)
        return pos


class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """

    def __init__(self, num_pos_feats=256):
        super().__init__()
        self.row_embed = nn.Embedding(50, num_pos_feats)
        self.col_embed = nn.Embedding(50, num_pos_feats)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.row_embed.weight)
        nn.init.uniform_(self.col_embed.weight)

    def forward(self, x):
        h, w = x.shape[-2:]
        i = torch.arange(w, device=x.device)
        j = torch.arange(h, device=x.device)
        x_emb = self.col_embed(i)
        y_emb = self.row_embed(j)
        pos = torch.cat([
            x_emb.unsqueeze(0).repeat(h, 1, 1),
            y_emb.unsqueeze(1).repeat(1, w, 1),
        ], dim=-1).permute(2, 0, 1).unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        return pos


def build_position_encoding(args):
    N_steps = args.hidden_dim // 2
    if args.position_embedding in ('v2', 'sine'):
        # TODO find a better way of exposing other arguments
        position_embedding = PositionEmbeddingSine(N_steps, normalize=True)
    elif args.position_embedding in ('v3', 'learned'):
        position_embedding = PositionEmbeddingLearned(N_steps)
    else:
        raise ValueError(f"not supported {args.position_embedding}")

    return position_embedding


