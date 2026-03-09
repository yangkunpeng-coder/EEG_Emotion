import pandas as pd


def get_channel_index(channel_name: str, path='../data/SEED/channel-order.xlsx'):
    """
    get channel index, the first one is 1
    :param path:
    :param channel_name:
    :return:
    """
    channel_s = pd.read_excel(path, header=None).values
    index = 0
    for channel in channel_s:
        index += 1
        if channel_name == channel:
            break
    assert (index != channel_s.shape[0])
    return index


if __name__ == '__main__':
    print(get_channel_index('FP1'))
