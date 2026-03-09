# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Emperor_Yang, Inc. All Rights Reserved 
#
# @CreateTime    : 2023/1/15 11:50
# @Author        : Emperor_Yang 
# @File          : edg_index.py
# @Software      : PyCharm
import os.path

import torch
import pandas as pd
import numpy as np
from utils.local_connect_matrix import get_local_connect_matrix


def build_local_edge_index_pt(path: str):
    """
    :param path:     'local_edge_index.pt'
    :return:
    """
    assert (os.path.exists(path))
    edge_index = torch.load(path)
    return edge_index


def build_local_edge_index_xlsx(path: str):
    """
    build edge from .xlsx file,for example '../data/SEED/local_connect__matrix.xlsx'
    :param path:
    :return:
    """
    data_df = pd.read_excel(path)
    data_df.fillna(0, inplace=True)
    data_np = data_df.values.astype(np.compat.long)
    edge_index_s = []
    for row in range(data_np.shape[0]):
        for col in range(data_np.shape[1]):
            if data_np[row][col] == 1:
                edge_index_s.append([row, col])
    edge_index = torch.tensor(edge_index_s, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    return edge_index


def build_local_edge_index_code():
    data_np = np.array(get_local_connect_matrix(), dtype=np.compat.long)
    edge_index_s = []
    for row in range(data_np.shape[0]):
        for col in range(data_np.shape[1]):
            if data_np[row][col] == 1:
                edge_index_s.append([row, col])
    edge_index = torch.tensor(edge_index_s, dtype=torch.long)
    edge_index = edge_index.t().contiguous()
    # torch.save(edge_index, 'local_edge_index.pt')
    return edge_index

