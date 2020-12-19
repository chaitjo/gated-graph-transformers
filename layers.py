import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
import dgl.function as fn


class MLPLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, 
                 dropout=0.0, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        # MLPs for node and edge features
        self.ff_h = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
        self.ff_e = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
        
        if batch_norm == True:
            self.norm_h = nn.BatchNorm1d(output_dim)
            self.norm_e = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):
        
        h_in = h  # for residual connection
        e_in = e  # for residual connection
        
        if self.batch_norm == True:
            h = self.norm_h(h)  # batch normalization  
            e = self.norm_e(e)  # batch normalization 
        
        # MLPs on updated node and edge features
        h = self.ff_h(h)
        e = self.ff_e(e)
        
        if self.residual == True:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection
        
        return h, e
    
    
class GCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, 
                 dropout=0.0, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        # Linear transformations
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        
        # MLPs for node and edge features
        self.ff_h = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
        self.ff_e = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
        
        if batch_norm == True:
            self.norm1_h = nn.BatchNorm1d(output_dim)
            self.norm2_h = nn.BatchNorm1d(output_dim)
            self.norm_e = nn.BatchNorm1d(output_dim)
            
    def forward(self, g, h, e):
        
        ########## Message-passing sub-layer ##########
        
        h_in = h  # for residual connection
        
        if self.batch_norm == True:
            h = self.norm1_h(h)  # batch normalization  
            
        # Linear transformations of nodes and edges
        g.ndata['h']  = h 
        g.ndata['Ah'] = self.A(h)  # node update, self-connection
        g.ndata['Bh'] = self.B(h)  # node update, neighbor projection
        
        # Graph convolution
        g.update_all(fn.copy_u('Bh', 'm'), fn.mean('m', 'neigh'))
        h = g.ndata['Ah'] + g.ndata['neigh']
        
        if self.residual == True:
            h = h_in + h  # residual connection
           
        ############ Feedforward sub-layer ############
        
        h_in = h  # for residual connection
        e_in = e  # for residual connection
        
        if self.batch_norm == True:
            h = self.norm2_h(h)  # batch normalization 
            e = self.norm_e(e)  # batch normalization 
        
        # MLPs on updated node and edge features
        h = self.ff_h(h)
        e = self.ff_e(e)
        
        if self.residual == True:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection
        
        return h, e


class GatedGCNLayer(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, 
                 dropout=0.0, batch_norm=True, residual=True, **kwargs):
        super().__init__()
        self.dropout = dropout
        self.batch_norm = batch_norm
        self.residual = residual
        
        if input_dim != output_dim:
            self.residual = False
        
        # Linear transformations for dense attention mechanism
        self.A = nn.Linear(input_dim, output_dim, bias=True)
        self.B = nn.Linear(input_dim, output_dim, bias=True)
        self.C = nn.Linear(input_dim, output_dim, bias=True)
        self.D = nn.Linear(input_dim, output_dim, bias=True)
        self.E = nn.Linear(input_dim, output_dim, bias=True)
        
        # MLPs for node and edge features
        self.ff_h = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
        self.ff_e = nn.Sequential(
            nn.Linear(output_dim, hidden_dim, bias=True),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim, bias=True)
        )
        
        if batch_norm == True:
            self.norm1_h = nn.BatchNorm1d(output_dim)
            self.norm1_e = nn.BatchNorm1d(output_dim)
            self.norm2_h = nn.BatchNorm1d(output_dim)
            self.norm2_e = nn.BatchNorm1d(output_dim)

    def forward(self, g, h, e):
        
        ########## Message-passing sub-layer ##########
        
        h_in = h  # for residual connection
        e_in = e  # for residual connection
        
        if self.batch_norm == True:
            h = self.norm1_h(h)  # batch normalization  
            e = self.norm1_e(e)  # batch normalization
        
        # Linear transformations of nodes and edges
        g.ndata['h']  = h 
        g.edata['e']  = e
        g.ndata['Ah'] = self.A(h)  # node update, self-connection
        g.ndata['Bh'] = self.B(h)  # node update, neighbor projection
        g.ndata['Ch'] = self.C(h)  # edge update, source node projection
        g.ndata['Dh'] = self.D(h)  # edge update, destination node projection
        g.edata['Ee'] = self.E(e)  # edge update, edge projection
        
        # Graph convolution with dense attention mechanism
        g.apply_edges(fn.u_add_v('Ch', 'Dh', 'CDh'))
        g.edata['e'] = g.edata['CDh'] + g.edata['Ee']
        # Dense attention mechanism
        g.edata['sigma'] = torch.sigmoid(g.edata['e'])
        g.update_all(fn.u_mul_e('Bh', 'sigma', 'm'), fn.sum('m', 'sum_sigma_h'))
        # Gated-Mean aggregation
        g.update_all(fn.copy_e('sigma', 'm'), fn.sum('m', 'sum_sigma'))
        g.ndata['h'] = g.ndata['Ah'] + g.ndata['sum_sigma_h'] / (g.ndata['sum_sigma'] + 1e-10)
        h = g.ndata['h']  # result of graph convolution
        e = g.edata['e']  # result of graph convolution 
        
        if self.residual == True:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection
        
        ############ Feedforward sub-layer ############
        
        h_in = h  # for residual connection
        e_in = e  # for residual connection
        
        if self.batch_norm == True:
            h = self.norm2_h(h)  # batch normalization  
            e = self.norm2_e(e)  # batch normalization 
        
        # MLPs on updated node and edge features
        h = self.ff_h(h)
        e = self.ff_e(e)
        
        if self.residual == True:
            h = h_in + h  # residual connection
            e = e_in + e  # residual connection
        
        return h, e

    
class GraphNorm(nn.Module):

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_dim))
        self.bias = nn.Parameter(torch.zeros(hidden_dim))
        self.mean_scale = nn.Parameter(torch.ones(hidden_dim))

    def forward(self, graph, tensor, batch_list):
        batch_size = len(batch_list)
        batch_list = batch_list.long().to(tensor.device)
        batch_index = torch.arange(batch_size).to(tensor.device).repeat_interleave(batch_list)
        batch_index = batch_index.view((-1,) + (1,) * (tensor.dim() - 1)).expand_as(tensor)
        
        mean = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        mean = mean.scatter_add_(0, batch_index, tensor)
        mean = (mean.T / batch_list).T
        mean = mean.repeat_interleave(batch_list, dim=0)

        sub = tensor - mean * self.mean_scale

        std = torch.zeros(batch_size, *tensor.shape[1:]).to(tensor.device)
        std = std.scatter_add_(0, batch_index, sub.pow(2))
        std = ((std.T / batch_list).T + 1e-6).sqrt()
        std = std.repeat_interleave(batch_list, dim=0)
        
        return self.weight * sub / std + self.bias


class SumPoolingEdges(nn.Module):
   
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata['e'] = feat
            readout = dgl.sum_edges(graph, 'e')
            return readout
        

class AvgPoolingEdges(nn.Module):
   
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata['e'] = feat
            readout = dgl.mean_edges(graph, 'e')
            return readout
    

class MaxPoolingEdges(nn.Module):
   
    def __init__(self):
        super().__init__()

    def forward(self, graph, feat):
        with graph.local_scope():
            graph.edata['e'] = feat
            readout = dgl.max_edges(graph, 'e')
            return readout
