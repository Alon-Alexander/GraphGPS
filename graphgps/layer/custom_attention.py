import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import to_dense_batch
from cdlib import algorithms
import networkx as nx


class CustomAttention(nn.Module):
    def __init__(self, 
                 dim_h: int, 
                 num_heads: int,
                 attn_dropout=0.0
                 ):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dim_h, num_heads, dropout=attn_dropout, batch_first=True)
    
    def forward(self, batch):
        h = batch.x
        print(batch)
        print(h.shape)
        G = to_networkx(batch, node_attrs=['x', 'batch'])
        print(G)

        coms = algorithms.walktrap(G)

        for community in coms.communities:
            subgraph = G.subgraph(community)

            torch_subgraph = from_networkx(subgraph)
            # print(torch_subgraph)

            # torch_subgraph.x *= 2

            for i, node in enumerate(subgraph.nodes):
                if G.nodes[node]['x'] != torch_subgraph.x[i].tolist():
                    print("HAHAHA", i, node)
                G.nodes[node]['x'] = torch_subgraph.x[i].tolist()

        data_after_communities = from_networkx(G)

        print("After")
        print(data_after_communities)
        print(torch.equal(data_after_communities.batch, batch.batch))
        print(torch.equal(data_after_communities.x, batch.x))

        h_dense, mask = to_dense_batch(
            data_after_communities.x, 
            data_after_communities.batch,
            )
        
        return self.attn.forward(h_dense, h_dense, h_dense, 
                                 attn_mask=None, 
                                 key_padding_mask=~mask, 
                                 need_weights=False,
                                 )[0][mask]
