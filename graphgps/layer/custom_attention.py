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
        # Transform to networkx graph
        G = to_networkx(batch, node_attrs=['x', 'batch'])

        # Separate into communities
        coms = algorithms.walktrap(G)

        for community in coms.communities:
            subgraph = G.subgraph(community)

            # Calculate attention inside community subgraph
            torch_subgraph = from_networkx(subgraph)
            dense_subgraph, dense_mask = to_dense_batch(
                torch_subgraph.x, 
                torch_subgraph.batch
                )
            torch_subgraph.x = self.attn.forward(
                dense_subgraph, dense_subgraph, dense_subgraph, 
                attn_mask=None, 
                key_padding_mask=~dense_mask, 
                need_weights=False,
            )[0][dense_mask]

            # Copy data back to networkx graph from community
            for i, node in enumerate(subgraph.nodes):
                G.nodes[node]['x'] = torch_subgraph.x[i].tolist()

        data_after_communities = from_networkx(G)

        return data_after_communities.x
        # return self.attn.forward(h_dense, h_dense, h_dense, 
        #                          attn_mask=None, 
        #                          key_padding_mask=~mask, 
        #                          need_weights=False,
        #                          )[0][mask]
