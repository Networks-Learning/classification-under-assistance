import os
import pickle
import numpy as np


def save(obj, output_file):
    with open(output_file + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def write_txt_file(data, file_name):
    with open(file_name, 'w') as f:
        for line in data:
            f.write(" ".join(map(str, line)) + '\n')


def load_data(input_file, flag=None):
    if flag == 'ifexists':
        if not os.path.isfile(input_file + '.pkl'):
            # print 'not found', input_file
            return {}
    with open(input_file + '.pkl', 'rb') as f:
        data = pickle.load(f)
    return data


def get_color_list():
    color_list = [(0.4, 0.7607843137254902, 0.6470588235294118),
                  (0.9882352941176471, 0.5529411764705883, 0.3843137254901961),
                  (0.5529411764705883, 0.6274509803921569, 0.796078431372549),
                  (0.9058823529411765, 0.5411764705882353, 0.7647058823529411),
                  (0.6509803921568628, 0.8470588235294118, 0.32941176470588235),
                  (1.0, 0.8509803921568627, 0.1843137254901961),
                  (0.8980392156862745, 0.7686274509803922, 0.5803921568627451),
                  (0.7019607843137254, 0.7019607843137254, 0.7019607843137254)]

    return color_list
