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

        self.virtualnode_emb = torch.nn.Embedding(1, emb_dim)
        torch.nn.init.constant_(self.virtualnode_emb.weight.data, 0)

        self.virtualnode_ff = nn.ModuleList([
            nn.Sequential(
                nn.BatchNorm1d(emb_dim),
                nn.Linear(emb_dim, hidden_dim, bias=True),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, emb_dim, bias=True)
            ) 
                for _ in range(num_layer - 1) 
        ])
        
        self.pooler_h = {
            "mean": AvgPooling(),
            "sum": SumPooling(),
            "max": MaxPooling(),
        }.get(graph_pooling, AvgPooling())
        
        self.pooler_e = {
            "mean": AvgPoolingEdges(),
            "sum": SumPoolingEdges(),
            "max": MaxPoolingEdges(),
        }.get(graph_pooling, AvgPoolingEdges())
        
        self.predictor = nn.Sequential(
            nn.Linear(2* emb_dim, hidden_dim, bias=True),
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
        
        # Initialize virtual node
        virtualnode = self.virtualnode_emb(torch.zeros(g.batch_size).long().to(h.device))
        batch_list = g.batch_num_nodes().long().to(h.device)
        batch_index = torch.arange(g.batch_size).long().to(h.device).repeat_interleave(batch_list)

        # Node and edge embeddings
        for layer_idx in range(self.num_layer):
            # Add message from virtual node to graph nodes
            h = h + virtualnode[batch_index]

            # Graph convolution
            h, e = self.layers[layer_idx](g, h, e)

            # Update virtual node
            if layer_idx < self.num_layer - 1:
                # Add message from graph nodes to virtual node
                virtualnode = virtualnode + self.pooler_h(g, h)
                virtualnode = self.virtualnode_ff[layer_idx](virtualnode)

        g.ndata['h'] = h
        g.edata['e'] = e
        
        # Graph embedding
        hg = torch.cat(
            (self.pooler_h(g, h), self.pooler_e(g, e)), 
            dim=-1
        )
        
        return self.predictor(hg)
