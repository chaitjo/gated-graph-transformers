import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

from layers import *


class GNN_mol(nn.Module):
    
    def __init__(self, gnn_type, num_tasks, 
                 num_layer=5, emb_dim=128, dropout=0.5, batch_norm=True, 
                 residual=True, pos_enc_dim=10, graph_pooling="mean"):
        super().__init__()
        
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        self.pos_enc_dim = pos_enc_dim
        self.graph_pooling = graph_pooling
        
        hidden_dim = 4* emb_dim
        self.hidden_dim = hidden_dim
        
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        
        if self.pos_enc_dim > 0:
            self.pos_encoder_h = nn.Linear(pos_enc_dim, emb_dim, bias=True)
        
        gnn_layer = {
            'gated-gcn': GatedGCNLayer,
            'gcn': GCNLayer,
            'mlp': MLPLayer,
        }.get(gnn_type, GatedGCNLayer)
         
        self.layers = nn.ModuleList([
            gnn_layer(emb_dim, emb_dim, 
                      hidden_dim=hidden_dim, dropout=dropout, 
                      batch_norm=batch_norm, residual=residual) 
                for _ in range(num_layer) 
        ])
        
        # self.pooler_h = {
        #     "mean": AvgPooling(),
        #     "sum": SumPooling(),
        #     "max": MaxPooling(),
        # }.get(graph_pooling, AvgPooling())
        
        # self.pooler_e = {
        #     "mean": AvgPoolingEdges(),
        #     "sum": SumPoolingEdges(),
        #     "max": MaxPoolingEdges(),
        # }.get(graph_pooling, AvgPoolingEdges())
        
        self.predictor = nn.Sequential(
            nn.Linear(6* emb_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_tasks, bias=True)
        )
        
    def forward(self, g, h, e):
        
        h = self.atom_encoder(h)
        e = self.bond_encoder(e)
        
        if self.pos_enc_dim > 0: # if 'pos_enc' in g.ntypes:
            # Add positional encodings
            pe_h = g.ndata['pos_enc'].to(h.device)
            # Add random sign flipping for PEs
            sign_flip = torch.randint(low=0, high=2, size=(1, pe_h.size(1)), device=pe_h.device)
            sign_flip[sign_flip==0.0] = -1
            pe_h = pe_h * sign_flip
            h = h + self.pos_encoder_h(pe_h)
        
        # Node and edge embeddings
        for conv in self.layers:
            h, e = conv(g, h, e)
        g.ndata['h'] = h
        g.edata['e'] = e
        
        # Graph embedding
        hg = torch.cat(
            (
                AvgPooling()(g, h),
                SumPooling()(g, h),
                MaxPooling()(g, h), 
                AvgPoolingEdges()(g, e),
                SumPoolingEdges()(g, e),
                MaxPoolingEdges()(g, e),
            ), 
            dim=-1
        )
        
        return self.predictor(hg)
