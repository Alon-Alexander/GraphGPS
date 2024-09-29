import gps_layer
from torch_geometric.datasets import TUDataset
from torch_geometric.loader import DataLoader
import torch


if __name__ == '__main__':
    print('my main')

    layer = gps_layer.GPSLayer(
        dim_h=21,
        local_gnn_type='None',
        global_model_type='CustomAttention',
        num_heads=3,
    )

    dataset = TUDataset(root='/tmp/ENZYMES', name='ENZYMES', use_node_attr=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)

    for i, batch in enumerate(loader):
        print(batch)

        result = layer.forward(batch)

        print(result)
        
        if i == 0:
            break


