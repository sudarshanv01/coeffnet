import torch
from torch.nn import Linear
from torch_geometric.nn import GraphConv
from torch_geometric.nn import global_mean_pool
import torch.nn.functional as F


class ClassifierModel(torch.nn.Module):
    def __init__(
        self, num_node_features, hidden_channels, out_channels, num_global_features=1
    ):
        super().__init__()

        self.num_global_features = num_global_features
        in_channels = num_node_features + num_global_features

        # Node feature embedding
        self.conv1 = GraphConv(in_channels, hidden_channels)
        self.conv2 = GraphConv(hidden_channels, hidden_channels)
        self.conv3 = GraphConv(hidden_channels, hidden_channels)
        self.lin = Linear(hidden_channels, out_channels)

    def forward(self, data):

        x, edge_index, edge_attr, u, batch = (
            data.x,
            data.edge_index,
            data.edge_attr,
            data.u,
            data.batch,
        )

        # Determine the edge_weight by the inverse of the bond length
        edge_weight = 1.0 / edge_attr
        # Make sure that any infinities are set to zero
        edge_weight[torch.isinf(edge_weight)] = 0.0

        # Add the global features to the node features
        u = u.view(-1, self.num_global_features)
        x = torch.cat([x, u[batch]], dim=1)

        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_weight=edge_attr**-1)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_weight=edge_attr**-1)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_weight=edge_attr**-1)

        # 3. Readout layer
        x = global_mean_pool(x, batch)

        # 4. Apply a final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin(x)

        return x
