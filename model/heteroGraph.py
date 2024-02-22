import torch
from torch import nn
from torch_geometric.nn import HGTConv, Linear

class HGT(nn.Module):

    def __init__(self, hidden_channels, out_channels, num_heads, num_layers, data):
        super(HGT, self).__init__()
        self.lin_dict = torch.nn.ModuleDict()
        for node_type in data.node_types:
            if(node_type == "text"):
                # text
                self.lin_dict[node_type] = Linear(512, hidden_channels)
            else:
                # audio/emotion/speaker/intensity
                self.lin_dict[node_type] = Linear(256, hidden_channels)


        self.convs = torch.nn.ModuleList()
        for _ in range(num_layers):
            conv = HGTConv(hidden_channels, hidden_channels, data.metadata(),
                           num_heads, group='sum')
            self.convs.append(conv)

        self.lin = Linear(hidden_channels, out_channels)


    def forward(self, x_dict, edge_index_dict):

        x_dict = {
            node_type: self.lin_dict[node_type](x).relu_()
            for node_type, x in x_dict.items()
        }

        for conv in self.convs:
            x_dict = conv(x_dict, edge_index_dict)

        # return self.lin(x_dict['people'])
        return x_dict["text"],x_dict["emotion"],x_dict["intensity"]

