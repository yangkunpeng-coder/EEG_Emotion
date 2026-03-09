import scipy.io


def load_m_data(path):
    """
    加载m数据
    :param path:
    :return:
    """
    data_dict = scipy.io.loadmat(path)
    # print(data_dict.keys())
    return data_dict


if __name__ == '__main__':
    load_m_data('../data/SEED/ExtractedFeatures/label.mat')
