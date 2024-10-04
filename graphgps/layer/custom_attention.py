import torch
import torch.nn as nn
from torch_geometric.utils.convert import to_networkx, from_networkx
from torch_geometric.utils import to_dense_batch
from cdlib import algorithms
import networkx as nx


def diagonalize_3d(matrices):
    shape = torch.Size([
        sum(mat.shape[0] for mat in matrices),
        sum(mat.shape[1] for mat in matrices),
        matrices[0].shape[2]
    ])

    out = torch.zeros(shape)
    
    i_cur = 0
    j_cur = 0
    for mat in matrices:
        i_size, j_size, *_ = mat.shape
        out[i_cur:i_cur + i_size, j_cur:j_cur + j_size, :] = mat
        i_cur += i_size
        j_cur += j_size

    return out


class CustomAttention(nn.Module):
    def __init__(self, 
                 dim_h: int, 
                 num_heads: int,
                 attn_dropout=0.0,
                 operate_individually: bool = True,
                 ):
        super().__init__()
        self.attn = torch.nn.MultiheadAttention(
            dim_h, num_heads, dropout=attn_dropout, batch_first=True)
        self.operate_individually = operate_individually
    
    def forward(self, batch):
        # Transform to networkx graph
        G = to_networkx(batch, node_attrs=['x', 'batch'])

        # Separate into communities
        coms = algorithms.walktrap(G)

        # For joining later
        dense_subgraphs = []
        dense_masks = []

        for community in coms.communities:
            subgraph = G.subgraph(community)

            # Calculate attention inside community subgraph
            torch_subgraph = from_networkx(subgraph)
            dense_subgraph, dense_mask = to_dense_batch(
                torch_subgraph.x, 
                torch_subgraph.batch
                )
            if self.operate_individually:
                torch_subgraph.x = self.attn.forward(
                    dense_subgraph, dense_subgraph, dense_subgraph, 
                    attn_mask=None, 
                    key_padding_mask=~dense_mask, 
                    need_weights=False,
                )[0][dense_mask]

                # Copy data back to networkx graph from community
                for i, node in enumerate(subgraph.nodes):
                    G.nodes[node]['x'] = torch_subgraph.x[i].tolist()

            else:
                dense_subgraphs.append(dense_subgraph)
                dense_masks.append(dense_mask)


        if self.operate_individually:
            data_after_communities = from_networkx(G)
            return data_after_communities.x
        else:
            joint_dense = diagonalize_3d(dense_subgraphs)
            joint_mask = torch.block_diag(*dense_masks)

            return self.attn.forward(joint_dense, joint_dense, joint_dense, 
                                    attn_mask=None, 
                                    key_padding_mask=~joint_mask, 
                                    need_weights=False,
                                    )[0][joint_mask]
