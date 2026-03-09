import os
import torch
from easydict import EasyDict
from torch.nn import ReLU
from torch_geometric.data import DataLoader
from torch_geometric.nn import DeepGCNLayer, global_add_pool, GCNConv, BatchNorm
from data_process.seed_loader_gnn_memory import SeedGnnMemoryDataset
from torch.utils.data import random_split
import collections

config = EasyDict()
config.learn_rate = 0.01
config.epoch = 50
config.note_feature_dim = 5
config.note_num = 62
config.hidden_channels = 16
config.class_num = 3
config.hidden_layers = 3
config.batch_size = 16
config.max_loss_increase_time = 3


class EEG_DeeperGCNs(torch.nn.Module):
    """
    GCN handle ECG
    """

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(EEG_DeeperGCNs, self).__init__()

        self.layers = torch.nn.ModuleList()
        conv = GCNConv(in_channels, hidden_channels, cached=True, normalize=True)
        self.layers.append(conv)
        for i in range(0, config.hidden_layers):
            conv = GCNConv(hidden_channels, hidden_channels, cached=True, normalize=True)
            norm = BatchNorm(hidden_channels)
            act = ReLU(inplace=True)
            layer = DeepGCNLayer(conv, norm, act, block='res+', dropout=0.1)
            self.layers.append(layer)

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
        for layer in self.layers:
            x = layer(x, edge_index, edge_weight)
        x = global_add_pool(x, index)
        x = self.fc1(x)
        return x


def train(model, optimizer, criterion, train_data_loader):
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


def test(model, test_data_loader):
    count = 0
    data_size = 0
    model.eval()
    with torch.no_grad():
        for mini_batch in test_data_loader:
            if mini_batch.num_graphs == config.batch_size:
                out = model(mini_batch.x, mini_batch.edge_index, mini_batch.batch)
                predict = torch.argmax(out, dim=1)
                count += int(predict.eq(mini_batch.y).sum())
                data_size += mini_batch.num_graphs
        # print("Test Accuracy:{}%".format(count / data_size * 100))
    return count / data_size * 100


def get_subjects_data_dict():
    """
    获取受试者所有的数据，按照字典进行存储
    :return:
    """
    data_dir = '../data/SEED/processed'
    name_s = os.listdir(data_dir)
    data_set_dict = dict()
    for name in name_s:
        subject_id = name.split('_')[0]
        if 'pre' in subject_id:
            continue
        subject_id = int(subject_id)
        part_data_set = SeedGnnMemoryDataset(root='../data/SEED/', processed_file=name)
        if subject_id in data_set_dict:
            data_set_dict[subject_id] += part_data_set
        else:
            data_set_dict[subject_id] = part_data_set
    return data_set_dict


if __name__ == '__main__':
    data_set_dict = get_subjects_data_dict()
    # subject dependent

    for subject in range(1, 16):
        # 构建模型、数据集、优化器、损失函数
        model = EEG_DeeperGCNs(config.note_feature_dim, config.hidden_channels, config.class_num)
        data_set = data_set_dict[subject]
        train_num = int(0.8 * len(data_set))
        test_num = len(data_set) - train_num
        train_data_set, test_data_set = random_split(data_set, [train_num, test_num])
        train_data_loader = DataLoader(train_data_set, batch_size=config.batch_size, shuffle=True)
        test_data_loader = DataLoader(test_data_set, batch_size=config.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # 训练和测试
        accuracy_s = []
        for epoch in range(config.epoch):
            avg_loss = train(model=model, optimizer=optimizer, criterion=criterion, train_data_loader=train_data_loader)
            if epoch > 15:
                accuracy_s.append(test(model, test_data_loader))
        print('subject dependent: subject_id = {}  max accuracy = {}'.format(subject, max(accuracy_s)))

    # subject independent
    for subject in range(1, 16):
        model = EEG_DeeperGCNs(config.note_feature_dim, config.hidden_channels, config.class_num)
        test_data_set = data_set_dict[subject]
        train_data_set_s = []
        for key in data_set_dict.keys():
            if key != subject:
                train_data_set_s.append(data_set_dict[key])
        train_data_set = train_data_set_s[0]
        for i in range(1, len(train_data_set_s)):
            train_data_set += train_data_set_s[i]
        train_data_loader = DataLoader(train_data_set, batch_size=config.batch_size, shuffle=True)
        test_data_loader = DataLoader(test_data_set, batch_size=config.batch_size, shuffle=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=config.learn_rate)
        criterion = torch.nn.CrossEntropyLoss()
        # 训练和测试
        accuracy_s = []
        for epoch in range(config.epoch):
            avg_loss = train(model=model, optimizer=optimizer, criterion=criterion, train_data_loader=train_data_loader)
        accuracy_s.append(test(model, test_data_loader))
        print('subject independent: subject_id = {}  max accuracy = {}'.format(subject, max(accuracy_s)))
