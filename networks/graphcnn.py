import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
sys.path.append("networks/")

import dgl
from dgl.nn import GATConv
from torch.nn.modules.module import Module

class GAT(Module):
    """backbone"""
    def __init__(self,
                 num_layers,
                 input_dim,
                 hidden_dim,
                 activation,
                 n_heads,
                 feat_drop=0.5,
                 attn_drop=0.5,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        self.gat_layers.append(GATConv(
            input_dim, hidden_dim, n_heads,
            feat_drop, attn_drop, negative_slope, residual, self.activation))
        # hidden layers
        for l in range(1, num_layers-1):
            # due to multi-head, the in_dim = num_hidden * n_heads
            self.gat_layers.append(GATConv(
                hidden_dim * n_heads, hidden_dim, n_heads,
                feat_drop, attn_drop, negative_slope, residual, self.activation))
        # output layer (n_heads = 1)
        self.gat_layers.append(GATConv(
            hidden_dim * n_heads, hidden_dim, 1,
            feat_drop, attn_drop, negative_slope, residual, self.activation))


    def forward(self, inputs: torch.Tensor, g: dgl.graph):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](g, h).flatten(1)
        
        return h


class SupConGraphCNN(Module):
    """backbone + projection head"""
    def __init__(self, num_graphcnn_layers, input_dim, hidden_dim, n_heads, head='mlp'):
        super(SupConGraphCNN, self).__init__()
        self.encoder = GAT(num_graphcnn_layers, input_dim, hidden_dim, F.elu, n_heads)
        if head == 'linear':
            self.head = nn.Linear(hidden_dim, hidden_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, feats, adj):
        emb = self.encoder(feats, adj)
        return F.normalize(self.head(emb), dim=1)

