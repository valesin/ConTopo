import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def get_grid_shape(n_units):
    """
    Choose (h, w) such that h * w == n_units and h is as close to w as possible.
    Returns a compact grid layout for arranging units.
    """
    h = int(math.sqrt(n_units))
    while n_units % h != 0:
        h -= 1
    w = n_units // h
    return h, w


def pos_dist(embedding_dim):
    """
    Build a pairwise Euclidean distance matrix D for a 2D grid with
    embedding_dim cells arranged as close to square as possible.
    D[i, j] is the distance between grid positions of units i and j.
    """
    h, w = get_grid_shape(embedding_dim)
    y = torch.linspace(0, 1, steps=h)
    x = torch.linspace(0, 1, steps=w)
    YY, XX = torch.meshgrid(y, x, indexing="ij")
    pos_hw2 = torch.stack([XX, YY], dim=-1)
    pos = pos_hw2.reshape(-1, 2)
    D = torch.cdist(pos, pos, p=2)
    return D


class Global_Topographic_Loss(nn.Module):
    """
    Global topographic regularizer on pre-activation features.
    Encourages cosine similarity between unit activations to decay with spatial distance:
      target_ij ≈ 1 / (d_ij + 1)
    where d_ij comes from a fixed grid-based distance matrix computed at init.
    """

    def __init__(self, weight=1.0, emb_dim=256):
        super(Global_Topographic_Loss, self).__init__()
        self.weight = weight
        self.D = pos_dist(emb_dim)

    def forward(self, pre_relu):
        if pre_relu is None:
            raise ValueError("pre_relu must be provided.")
        if pre_relu.dim() != 2:
            raise ValueError(
                f"pre_relu must be 2D [B, C], got shape {tuple(pre_relu.shape)}"
            )

        # Keep D on the same device as inputs
        self.D = self.D.to(pre_relu.device)

        _, n_units = pre_relu.shape

        if self.D.shape != (n_units, n_units):
            raise ValueError(
                f"D must have shape ({n_units}, {n_units}), got {tuple(self.D.shape)}"
            )

        # Cosine similarity across units (columns)
        Xn = F.normalize(
            pre_relu, p=2, dim=0, eps=1e-12
        )  # [B, C], L2-normalize per unit
        S = Xn.t() @ Xn  # (C, C) cosine sim matrix

        # Use upper triangle (i<j) to avoid double-counting/self-pairs
        i_idx, j_idx = torch.triu_indices(
            n_units, n_units, offset=1, device=pre_relu.device
        )
        d = self.D[i_idx, j_idx]
        s = S[i_idx, j_idx]

        # Quadratic penalty towards 1/(d+1); average over unordered pairs
        topo_loss_val = ((s - (1.0 / (d + 1.0))) ** 2).sum()
        return self.weight * (2.0 / (n_units * (n_units - 1))) * topo_loss_val


def grid_diffs(W):
    out_feats, in_feats = W.shape
    h, w = get_grid_shape(out_feats)
    G = W.reshape(h, w, in_feats)
    diffs = []
    if w > 1:
        diffs.append(G[:, :-1, :] - G[:, 1:, :])
    if h > 1:
        diffs.append(G[:-1, :, :] - G[1:, :, :])
    if h > 1 and w > 1:
        diffs.append(G[:-1, :-1, :] - G[1:, 1:, :])
        diffs.append(G[:-1, 1:, :] - G[1:, :-1, :])
    return diffs


# Add wrap around for torus topology
def torus_diffs(W):
    out_feats, in_feats = W.shape
    h, w = get_grid_shape(out_feats)
    G = W.reshape(h, w, in_feats)
    diffs = []
    if w > 1:
        diffs.append(G[:, :-1, :] - G[:, 1:, :])
        diffs.append(G[:, -1, :] - G[:, 0, :])
    if h > 1:
        diffs.append(G[:-1, :, :] - G[1:, :, :])
        diffs.append(G[-1, :, :] - G[0, :, :])
    if h > 1 and w > 1:
        diffs.append(G[:-1, :-1, :] - G[1:, 1:, :])
        diffs.append(G[:-1, 1:, :] - G[1:, :-1, :])
        diffs.append(G[-1, :-1, :] - G[0, 1:, :])
        diffs.append(G[-1, 1:, :] - G[0, :-1, :])
    return diffs


class Local_WS_Loss(nn.Module):
    """
    Local weight-smoothing regularizer for a linear layer.
    Arrange output units on a grid and penalize differences between
    neighboring rows of the weight matrix (right/down/diagonals).
    """

    def __init__(self, weight=1.0, topology="grid"):
        super(Local_WS_Loss, self).__init__()
        self.weight = weight
        self.topology = topology

        if topology == "grid":
            self.diff_fn = grid_diffs
        elif topology == "torus":
            self.diff_fn = torus_diffs
        else:
            raise ValueError(f"Unknown topology: {topology}")

    def forward(self, linear_layer=None):
        if linear_layer is None:
            raise ValueError("linear_layer must be provided.")

        if not isinstance(linear_layer, nn.Linear):
            raise ValueError("linear_layer must be an instance of nn.Linear.")

        W = linear_layer.weight
        diffs = self.diff_fn(W)

        if not diffs:
            return torch.zeros((), device=W.device, dtype=W.dtype)

        # L2 over feature dim, then mean over all neighbor pairs
        dists = [torch.linalg.norm(d, dim=-1) for d in diffs]
        topo_loss_val = torch.cat([x.reshape(-1) for x in dists]).mean()

        return self.weight * topo_loss_val
