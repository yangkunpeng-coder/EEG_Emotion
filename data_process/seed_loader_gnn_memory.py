# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Emperor_Yang, Inc. All Rights Reserved 
#
# @CreateTime    : 2023/1/14 22:38
# @Author        : Emperor_Yang 
# @File          : seed_loader_gnn.py
# @Software      : PyCharm

import os
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.data import InMemoryDataset
from data_process.edge_index import build_local_edge_index_pt
from data_process.feature_x import build_graph_feature_data
from torch_geometric.loader import DataLoader


class SeedGnnMemoryDataset(InMemoryDataset):
    def __init__(self, root, processed_file, transform=None, pre_transform=None, pre_filter=None):
        super().__init__(root, transform, pre_transform, pre_filter)
        path = os.path.join(self.processed_dir, processed_file)
        assert (path in self.processed_paths)
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        file_name_s = []
        data_dir = os.path.join(self.root, 'ExtractedFeatures/')
        for file_name in os.listdir(data_dir):
            if 'label.mat' in file_name:
                continue
            if '.mat' in file_name:
                file_name_s.append(file_name)
        # 规定将label路径放在最后一个
        file_name_s.append('label.mat')
        return file_name_s

    @property
    def processed_file_names(self):
        processed_files = []
        for file_name in self.raw_file_names[:-1]:
            processed_files.append(file_name.replace('.mat', '.pt'))
        return processed_files

    def download(self):
        ...

    def process(self):
        extracted_features_dir = os.path.join(self.root, 'ExtractedFeatures/')
        edge_index = build_local_edge_index_pt(os.path.join(self.root, 'local_edge_index.pt'))

        # 迭代文件列表，对每个文件进行处理，得到图数据
        for i, file_name in enumerate(self.raw_file_names[:-1]):
            data_list = []
            feature_path = os.path.join(extracted_features_dir, file_name)
            label_path = os.path.join(extracted_features_dir, 'label.mat')
            # 构建图的Data对象,放到列表中
            data_x_s, label_s = build_graph_feature_data(feature_path, label_path)  # data_x : (samples, 62, 5)
            label_s = [label + 1 for label in label_s]
            # 迭代样本，对每个样本进行处理，构建Data格式
            for sample_index in range(data_x_s.shape[0]):
                x_list = []
                for channel_index in range(data_x_s.shape[1]):
                    x_list.append(data_x_s[sample_index, channel_index, :])
                x = torch.tensor(np.array(x_list), dtype=torch.float)
                one_data = Data(x=x, edge_index=edge_index, y=torch.tensor(label_s[sample_index], dtype=torch.long))
                data_list.append(one_data)

            # 进行数据过滤
            if self.pre_filter is not None:
                data_list = [data for data in data_list if self.pre_filter(data)]

            # 进行数据预转换
            if self.pre_transform is not None:
                data_list = [self.pre_transform(data) for data in data_list]

            # 对预转换的数据进行压缩和保存
            data, slices = self.collate(data_list)
            torch.save((data, slices), self.processed_paths[i])


if __name__ == '__main__':
    data_set = SeedGnnMemoryDataset(root='../data/SEED/', processed_file='1_20131027.pt')
    data_loader = DataLoader(data_set, batch_size=32, shuffle=False, num_workers=8)
