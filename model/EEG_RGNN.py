from abc import ABC, ABCMeta
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from torch_geometric.nn import SGConv, global_add_pool
from torch_scatter import scatter_add
from easydict import EasyDict
from torch_geometric.data import DataLoader
from data_process.seed_loader_gnn_memory import SeedGnnMemoryDataset

config = EasyDict()
config.learn_rate = 0.01
config.epoch = 50
config.note_feature_dim = 5
config.note_num = 62
config.hidden_channels = 16
config.class_num = 3
config.hidden_layers = 30
config.batch_size = 16
config.max_loss_increase_time = 3
config.learn_edge_weight = True
config.K = 5


def maybe_num_nodes(index, num_nodes=None):
    return index.max().item() + 1 if num_nodes is None else num_nodes


def add_remaining_self_loops(edge_index,
                             edge_weight=None,
                             fill_value=1,
                             num_nodes=None):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    row, col = edge_index

    mask = row != col
    inv_mask = ~mask
    loop_weight = torch.full(
        (num_nodes,),
        fill_value,
        dtype=None if edge_weight is None else edge_weight.dtype,
        device=edge_index.device)

    if edge_weight is not None:
        assert edge_weight.numel() == edge_index.size(1)
        remaining_edge_weight = edge_weight[inv_mask]
        if remaining_edge_weight.numel() > 0:
            loop_weight[row[inv_mask]] = remaining_edge_weight
        edge_weight = torch.cat([edge_weight[mask], loop_weight], dim=0)

    loop_index = torch.arange(0, num_nodes, dtype=row.dtype, device=row.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    edge_index = torch.cat([edge_index[:, mask], loop_index], dim=1)

    return edge_index, edge_weight


class RSGConv(SGConv, metaclass=ABCMeta):
    def __init__(self, num_features, out_channels, K=1, cached=False, bias=True):
        super(RSGConv, self).__init__(num_features, out_channels, K=K, cached=cached, bias=bias)

    # allow negative edge weights
    @staticmethod
    def norm(edge_index, num_nodes, edge_weight, improved=False, dtype=None):
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1),),
                                     dtype=dtype,
                                     device=edge_index.device)

        fill_value = 1 if not improved else 2
        edge_index, edge_weight = add_remaining_self_loops(
            edge_index, edge_weight, fill_value, num_nodes)
        row, col = edge_index
        deg = scatter_add(torch.abs(edge_weight), row, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

    def forward(self, x, edge_index, edge_weight=None):
        """"""
        if not self.cached or self.cached_result is None:
            edge_index, norm = RSGConv.norm(
                edge_index, x.size(0), edge_weight, dtype=x.dtype)

            for k in range(self.K):
                x = self.propagate(edge_index, x=x, norm=norm)
        return self.lin(x)

    def message(self, x_j, norm):
        # x_j: (batch_size*num_nodes*num_nodes, num_features)
        # norm: (batch_size*num_nodes*num_nodes, )
        return norm.view(-1, 1) * x_j


class SymSimGCNNet(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels,
                 edge_weight=None, dropout=0.5):
        """
            edge_weight: initial edge matrix
            dropout: dropout rate in final linear layer
        """
        super(SymSimGCNNet, self).__init__()
        self.num_nodes = config.note_num
        self.xs, self.ys = torch.tril_indices(self.num_nodes, self.num_nodes, offset=0)
        if edge_weight is not None:
            edge_weight = edge_weight.reshape(self.num_nodes, self.num_nodes)[
                self.xs, self.ys]  # strict lower triangular values
            self.edge_weight = nn.Parameter(edge_weight, requires_grad=config.learn_edge_weight)
        else:
            self.edge_weight = None
        self.dropout = dropout

        self.conv_s = torch.nn.ModuleList()
        self.conv_s.append(RSGConv(num_features=in_channels, out_channels=hidden_channels, K=config.K))
        for i in range(config.hidden_layers - 1):
            self.conv_s.append(RSGConv(num_features=hidden_channels, out_channels=hidden_channels, K=config.K))

        self.fc = nn.Linear(hidden_channels, out_channels)

    def forward(self, data):
        batch_size = len(data.y)
        x, edge_index = data.x, data.edge_index
        edge_weight = None
        if self.edge_weight is not None:
            edge_weight = torch.zeros((self.num_nodes, self.num_nodes), device=edge_index.device)
            edge_weight[self.xs.to(edge_weight.device), self.ys.to(edge_weight.device)] = self.edge_weight
            edge_weight = edge_weight + edge_weight.transpose(1, 0) - torch.diag(
                edge_weight.diagonal())  # copy values from lower tri to upper tri
            edge_weight = edge_weight.reshape(-1).repeat(batch_size)

        for conv in self.conv_s:
            x = conv(x, edge_index, edge_weight).relu()
        x = global_add_pool(x, data.batch)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc(x)
        return x


model = SymSimGCNNet(config.note_feature_dim, config.hidden_channels, config.class_num)
data_set = SeedGnnMemoryDataset(root='../data/SEED/', processed_file='1_20131027.pt')
train_data_set = data_set[: int(0.8 * data_set.len())]
test_data_set = data_set[int(0.8 * data_set.len()):]
train_data_loader = DataLoader(train_data_set, batch_size=config.batch_size, shuffle=True)
test_data_loader = DataLoader(test_data_set, batch_size=config.batch_size, shuffle=False)
optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
criterion = torch.nn.CrossEntropyLoss()


def train():
    loss_sum = 0
    data_size = 0
    for mini_batch in train_data_loader:
        if mini_batch.num_graphs == config.batch_size:
            data_size += mini_batch.num_graphs
            model.train()
            optimizer.zero_grad()
            out = model(mini_batch)
            loss = criterion(out, mini_batch.y)
            loss.backward()
            optimizer.step()
            loss_sum += loss.item() / mini_batch.num_graphs
    return loss_sum / data_size


def test():
    count = 0
    data_size = 0
    for mini_batch in test_data_loader:
        if mini_batch.num_graphs == config.batch_size:
            out = model(mini_batch)
            predict = torch.argmax(out, dim=1)
            count += int(predict.eq(mini_batch.y).sum())
            data_size += mini_batch.num_graphs
    print("Test Accuracy:{}%".format(count / data_size * 100))
    return count / data_size * 100


if __name__ == '__main__':
    all_accuracy = []
    for epoch in range(config.epoch):
        avg_loss = train()
        # print("epoch:{}, loss:{}".format(epoch+1, avg_loss))
        if epoch > 10:
            accuracy = test()
            all_accuracy.append(accuracy)
    print('max Accuracy:{}%'.format(max(all_accuracy)))