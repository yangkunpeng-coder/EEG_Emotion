# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Emperor_Yang, Inc. All Rights Reserved 
#
# @CreateTime    : 2023/1/17 21:39
# @Author        : Emperor_Yang 
# @File          : edge_weight.py
# @Software      : PyCharm
import numpy as np
from utils.load_channel_index import get_channel_index
from utils.local_connect_matrix import get_local_connect_matrix


def build_edge_weight_DGCNN(dist_ij_2D: np.array) -> np.array:
    """
     paper:《EEG Emotion Recognition Using Dynamical Graph Convolutional Neural Networks》
    :param dist_ij_2D: SEED-EED distance matrix
    :return:
    """
    node_num = 62
    tau_value = 2  # What value does the paper not say
    theta_value = 2  # What value does the paper not say

    edge_weight = np.zeros((node_num, node_num), dtype=np.float)
    for i in range(node_num):
        for j in range(node_num):
            dist_ij = dist_ij_2D[i, j]
            edge_weight[i, j] = 0 if dist_ij > tau_value else np.exp(- dist_ij ** 2 / 2 * theta_value ** 2)
    return edge_weight


def build_edge_weight_RGNN(dist_ij_2D: np.array) -> np.array:
    """
     paper:《EEG-Based Emotion Recognition Using Regularized Graph Neural Networks》
    :param dist_ij_2D: SEED-EED distance matrix
    :return:
    """
    node_num = 62
    delta_value = 2
    global_connect_pair = [['FP1', 'FP2'],
                           ['AF3', 'AF4'],
                           ['F5', 'F6'],
                           ['FC5', 'FC6'],
                           ['C5', 'C6'],
                           ['CP5', 'CP6'],
                           ['P5', 'P6'],
                           ['PO5', 'PO6'],
                           ['O1', 'O2']]

    edge_weight = np.zeros((node_num, node_num), dtype=np.float)
    for i in range(node_num):
        for j in range(node_num):
            dist_ij = dist_ij_2D[i, j]
            edge_weight[i, j] = np.min(1, delta_value / dist_ij ** 2)
    for pair in global_connect_pair:
        i = get_channel_index(pair[0])
        j = get_channel_index(pair[1])
        edge_weight[i, j] = edge_weight[i, j] - 1
    return edge_weight


def build_edge_weight_equal(dist_ij_2D: np.array) -> np.array:
    """
    As long as they're connected, the weights are equal 1
    :param dist_ij_2D:
    :return:
    """
    edge_weight = np.array(get_local_connect_matrix())
    return edge_weight
