from __future__ import annotations

import torch

from src.models.lightgcn import LightGCN, build_norm_adj


def test_lightgcn_ratings_shape_on_tiny_graph():
    n_users = 3
    n_items = 4
    train_u = torch.tensor([0, 1, 2], dtype=torch.long)
    train_i = torch.tensor([0, 1, 2], dtype=torch.long)

    model = LightGCN(n_users=n_users, n_items=n_items, embed_dim=8, n_layers=1)
    model.set_graph(build_norm_adj(train_u, train_i, n_users, n_items, device="cpu"))

    ratings = model.get_all_ratings(torch.tensor([0, 2], dtype=torch.long))

    assert ratings.shape == (2, n_items)
    assert torch.isfinite(ratings).all()
