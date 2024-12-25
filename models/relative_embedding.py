import torch
import math
import numpy as np

def BoxRelationalEmbedding(f_g, dim_g=64, wave_len=1000, trignometric_embedding=True):
    """
    Given a tensor with bbox coordinates for detected objects on each batch image,
    this function computes a matrix for each image

    with entry (i,j) given by a vector representation of the
    displacement between the coordinates of bbox_i, and bbox_j

    input: np.array of shape=(batch_size, max_nr_bounding_boxes, 4)
    output: np.array of shape=(batch_size, max_nr_bounding_boxes, max_nr_bounding_boxes, 64)
    """
    # returns a relational embedding for each pair of bboxes, with dimension = dim_g
    # follow implementation of https://github.com/heefe92/Relation_Networks-pytorch/blob/master/model.py#L1014-L1055

    batch_size = f_g.size(0)
    x_min, y_min, x_max, y_max = torch.chunk(f_g, 4, dim=-1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # cx.view(1,-1) transposes the vector cx, and so dim(delta_x) = (dim(cx), dim(cx))
    delta_x = cx - cx.view(batch_size, 1, -1)
    delta_x = torch.clamp(torch.abs(delta_x / w), min=1e-3)
    delta_x = torch.log(delta_x)

    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.clamp(torch.abs(delta_y / h), min=1e-3)
    delta_y = torch.log(delta_y)

    delta_w = torch.log(w / w.view(batch_size, 1, -1))
    delta_h = torch.log(h / h.view(batch_size, 1, -1))

    matrix_size = delta_h.size()
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_w = delta_w.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_h = delta_h.view(batch_size, matrix_size[1], matrix_size[2], 1)

    position_mat = torch.cat((delta_x, delta_y, delta_w, delta_h), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        feat_range = torch.arange(dim_g / 8).cuda()
        dim_mat = feat_range / (dim_g / 8)
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        dim_mat = dim_mat.view(1, 1, 1, -1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        mul_mat = position_mat * dim_mat
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)

def GridRelationalEmbedding(batch_size, grid_size=7, dim_g=64, wave_len=1000, trignometric_embedding=True):
    a = torch.arange(0, grid_size).float().cuda()
    c1 = a.view(-1, 1).expand(-1, grid_size).contiguous().view(-1)
    c2 = a.view(1, -1).expand(grid_size, -1).contiguous().view(-1)
    c3 = c1 + 1
    c4 = c2 + 1
    f = lambda x: x.view(1, -1, 1).expand(batch_size, -1, -1)
    x_min, y_min, x_max, y_max = f(c1), f(c2), f(c3), f(c4)

    # (bs, 49, 1)
    cx = (x_min + x_max) * 0.5
    cy = (y_min + y_max) * 0.5
    w = (x_max - x_min) + 1.
    h = (y_max - y_min) + 1.

    # smoothing transformation
    delta_x = cx - cx.view(batch_size, 1, -1)  # (bs, 49, 49)
    delta_x = torch.sign(delta_x) * torch.log(1 + torch.abs(delta_x / w))
    delta_y = cy - cy.view(batch_size, 1, -1)
    delta_y = torch.sign(delta_y) * torch.log(1 + torch.abs(delta_y / h))

    # (bs, 49, 49)
    matrix_size = delta_x.size()
    # (bs, 49, 49) -> (bs, 49, 49, 1)
    delta_x = delta_x.view(batch_size, matrix_size[1], matrix_size[2], 1)
    delta_y = delta_y.view(batch_size, matrix_size[1], matrix_size[2], 1)
    dist = torch.sqrt(delta_x ** 2 + delta_y ** 2)
    # angle
    angle = torch.atan2(delta_y, delta_x)

    position_mat = torch.cat((delta_x, delta_y, dist, angle), -1)  # bs * r * r *4

    if trignometric_embedding == True:
        # (0,1,...,7)
        feat_range = torch.arange(dim_g / 8).cuda()
        # (0,1/8,...,7/8)
        dim_mat = feat_range / (dim_g / 8)
        # 1/(1000^(i/8))
        dim_mat = 1. / (torch.pow(wave_len, dim_mat))

        # (8,) -> (1, 1, 1, 8)
        dim_mat = dim_mat.view(1, 1, 1, -1)
        # (bs, 49, 49, 4, 1)
        position_mat = position_mat.view(batch_size, matrix_size[1], matrix_size[2], 4, -1)
        position_mat = 100. * position_mat

        # dim_mat: (1, 1, 1, 8) ->(b_s, 49, 49, 4, 8)
        mul_mat = position_mat * dim_mat
        # (b_s, 49, 49, 32)
        mul_mat = mul_mat.view(batch_size, matrix_size[1], matrix_size[2], -1)
        sin_mat = torch.sin(mul_mat)
        cos_mat = torch.cos(mul_mat)
        # dim_mat: (1, 1, 1, 8) ->(b_s, 49, 49, 64)
        embedding = torch.cat((sin_mat, cos_mat), -1)
    else:
        embedding = position_mat
    return (embedding)

