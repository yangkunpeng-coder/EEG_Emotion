# -*- coding: utf-8 -*-
#
# Copyright (C) 2022 Emperor_Yang, Inc. All Rights Reserved 
#
# @CreateTime    : 2023/1/15 13:57
# @Author        : Emperor_Yang 
# @File          : feature_x.py
# @Software      : PyCharm

from utils.load_m_data import load_m_data
import numpy as np


def build_ML_feature_data(feature_path: str, label_path: str):
    """
    build traditional Machine learning data format
    :param feature_path:
    :param label_path:
    :return:
    """
    # 基本信息
    feature_dict = load_m_data(feature_path)
    label_dict = load_m_data(label_path)
    subject_num = 15
    channel_num = 62
    frequency_band_num = 5

    # 求样本数
    sample_num_s = []
    sample_num_sum = 0
    for i in range(subject_num):
        feature_3d = feature_dict['de_LDS' + str(i+1)]
        sample_slice_num = feature_3d.shape[1]
        sample_num_s.append(sample_slice_num)
        sample_num_sum += sample_slice_num

    # 构造特征向量和标签,按照(samples, [channels_1, ..., channels_5])
    all_index = 0
    feature_all = np.zeros((sample_num_sum, channel_num * frequency_band_num))
    label_all = np.zeros((sample_num_sum, 1))
    for subject_index in range(subject_num):
        feature_3d = feature_dict['de_LDS' + str(subject_index+1)]
        sample_slice_num = feature_3d.shape[1]
        for sample_index in range(sample_slice_num):
            for band_index in range(frequency_band_num):
                for channel_index in range(channel_num):
                    feature_all[all_index + sample_index, frequency_band_num * band_index + channel_index] \
                        = feature_3d[channel_index, sample_index, band_index]
            label_all[all_index + sample_index] = label_dict['label'][0, subject_index]
        all_index += sample_num_s[subject_index]

    return feature_all, label_all


def build_graph_feature_data(feature_path: str, label_path: str):
    """
    build graph NN data format
    :param feature_path:
    :param label_path:
    :return:
    """
    # 基本信息
    feature_dict = load_m_data(feature_path)
    label_dict = load_m_data(label_path)
    subject_num = 15
    channel_num = 62
    frequency_band_num = 5

    # 求样本数
    sample_num_s = []
    sample_num_sum = 0
    for i in range(subject_num):
        feature_3d = feature_dict['de_LDS' + str(i+1)]
        sample_slice_num = feature_3d.shape[1]
        sample_num_s.append(sample_slice_num)
        sample_num_sum += sample_slice_num

    # 构造特征向量和标签,按照(samples, [node_1], ..., [node_62])  shape:(sample, 62, 5)
    all_index = 0
    feature_all = np.zeros((sample_num_sum, channel_num, frequency_band_num))
    label_all = np.zeros((sample_num_sum, 1), dtype=int)
    for subject_index in range(subject_num):
        feature_3d = feature_dict['de_LDS' + str(subject_index+1)]
        current_sample_num = feature_3d.shape[1]
        for sample_index in range(current_sample_num):
            for band_index in range(frequency_band_num):
                for channel_index in range(channel_num):
                    feature_all[all_index + sample_index, channel_index, band_index] \
                        = feature_3d[channel_index, sample_index, band_index]
            label_all[all_index + sample_index] = label_dict['label'][0, subject_index]
        all_index += sample_num_s[subject_index]
    # print(label_all)
    # print(feature_all)
    return feature_all, label_all


if __name__ == '__main__':
    feature_path_g = '../data/SEED/ExtractedFeatures/1_20131027.mat'
    label_path_g = '../data/SEED/ExtractedFeatures/label.mat'
    build_graph_feature_data(feature_path_g, label_path_g)
