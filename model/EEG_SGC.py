import torch
import torch.nn.functional as F
from easydict import EasyDict
from torch_geometric.nn import SGConv, global_add_pool
from torch_geometric.data import DataLoader
from data_process.seed_loader_gnn_memory import SeedGnnMemoryDataset

config = EasyDict()
config.learn_rate = 0.01
config.epoch = 20
config.note_feature_dim = 5
config.note_num = 62
config.hidden_channels = 16
config.class_num = 3
config.hidden_layers = 4
config.batch_size = 16
config.max_loss_increase_time = 3


class EEG_SGC(torch.nn.Module):
    """
    GCN handle ECG
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EEG_SGC, self).__init__()

        self.conv_s = torch.nn.ModuleList()
        self.conv_s.append(SGConv(in_channels, hidden_channels, cached=True, normalize=True))
        for i in range(config.hidden_layers - 1):
            self.conv_s.append(SGConv(hidden_channels, hidden_channels, cached=True, normalize=True))

        self.fc1 = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, edge_index, index, edge_weight=None):
        """
        forward
        :param index:
        :param x:note feature
        :param edge_index:edge pair
        :param edge_weight: edge feature
        :return:
        """
        for conv in self.conv_s:
            x = conv(x, edge_index, edge_weight).relu()
        x = global_add_pool(x, index)
        x = self.fc1(x)
        return x


model = EEG_SGC(config.note_feature_dim, config.hidden_channels, config.class_num)
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
            out = model(mini_batch.x, mini_batch.edge_index, mini_batch.batch)
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
            out = model(mini_batch.x, mini_batch.edge_index, mini_batch.batch)
            predict = torch.argmax(out, dim=1)
            count += int(predict.eq(mini_batch.y).sum())
            data_size += mini_batch.num_graphs
    print("Test Accuracy:{}%".format(count / data_size * 100))


if __name__ == '__main__':
    loss_increase_time = 0
    last_lost = 1
    for epoch in range(config.epoch):
        avg_loss = train()
        print("epoch:{}, loss:{}".format(epoch+1, avg_loss))
        if avg_loss > last_lost:
            loss_increase_time += 1
        else:
            last_lost = avg_loss
        # 如果连续增加loss大于config.max_loss_increase_time，则停止训练
        if loss_increase_time > config.max_loss_increase_time:
            break
    test()
