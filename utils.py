import torch
import dgl

import numpy as np
from scipy import sparse as sp


def add_positional_encoding(g, pos_enc_dim, add_edge_pe=False):
    # Graph positional encoding via Laplacian eigenvectors
    
    # Compute graph Laplacian
    A = g.adjacency_matrix_scipy(return_edge_ids=False).astype(float)
    N = sp.diags(dgl.backend.asnumpy(g.in_degrees()).clip(1) ** -0.5, dtype=float)
    L = sp.eye(g.number_of_nodes()) - N * A * N

    # Eigenvectors with numpy
    EigVal, EigVec = np.linalg.eig(L.toarray())
    idx = EigVal.argsort()  # sort in increasing order
    EigVal, EigVec = EigVal[idx], np.real(EigVec[:, idx])
    
    # Assign PEs
    if g.number_of_nodes() > pos_enc_dim + 1:
        pos_enc = torch.from_numpy(EigVec[:, 1:pos_enc_dim+1]).float() 
    else:
        # Add dummy zeros for graphs with number of nodes < PE dimension
        pos_enc = torch.zeros(g.number_of_nodes(), pos_enc_dim).float()
        pos_enc[:, :g.number_of_nodes()-1] = torch.from_numpy(EigVec[:, 1:]).float() 
    
    g.ndata['pos_enc'] = pos_enc
    if add_edge_pe == True:
        g.edata['pos_enc'] = torch.cat(
            (g.ndata['pos_enc'][g.edges()[0]], g.ndata['pos_enc'][g.edges()[1]]), 
            dim=-1
        )
    return g


def collate_dgl(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    labels = torch.stack(labels)
    return batched_graph, labels
