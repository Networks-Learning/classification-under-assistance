import sys
from myutil import *
import numpy as np
import numpy.random as rand
import numpy.linalg as LA
import random
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from random import seed, shuffle


class generate_data:
    def __init__(self, n, dim, list_of_std, std_y=None):
        self.n = n
        self.dim = dim
        self.list_of_std = list_of_std
        self.std_y = std_y

    def c_gaussian_linear_generate_data(self):
        def gen_gaussian(mean_in, cov_in, class_label, num):
            nv = multivariate_normal(mean=mean_in, cov=cov_in)
            X = nv.rvs(num)
            y = np.ones(num, dtype=float) * class_label
            return nv, X, y

        # We will generate one gaussian cluster for each class
        mu1, sigma1 = [4, 2], [[6, 1], [1, 10]]
        mu2, sigma2 = [-4, -2], [[6, 1], [1, 10]]

        nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1, self.n / 2)  # positive class
        nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1, self.n / 2)

        # join the posisitve and negative class clusters
        self.X = np.vstack((X1, X2))
        self.Y = np.hstack((y1, y2))

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

        # shuffle the data
        perm = range(0, self.n)
        shuffle(perm)
        self.X = self.X[perm]
        self.Y = self.Y[perm]

    def c_gaussian_kernel_generate_data(self):
        def gen_gaussian(mean_in, cov_in, class_label, num):
            nv = multivariate_normal(mean=mean_in, cov=cov_in)
            X = nv.rvs(num)
            y = np.ones(num, dtype=float) * class_label
            return nv, X, y

        mu1, sigma1 = [0, 0], [[14, 1], [1, 12]]
        mu2, sigma2 = [5, 5], [[8, 1], [1, 8]]
        mu3, sigma3 = [5, -5], [[8, 1], [1, 8]]
        mu4, sigma4 = [-5, 5], [[8, 1], [1, 8]]
        mu5, sigma5 = [-5, -5], [[8, 1], [1, 8]]

        nv1, X1, y1 = gen_gaussian(mu1, sigma1, 1, self.n / 2)
        nv2, X2, y2 = gen_gaussian(mu2, sigma2, -1, self.n / 8)
        nv3, X3, y3 = gen_gaussian(mu3, sigma3, -1, self.n / 8)
        nv4, X4, y4 = gen_gaussian(mu4, sigma4, -1, self.n / 8)
        nv5, X5, y5 = gen_gaussian(mu5, sigma5, -1, self.n / 8)

        self.X = np.vstack((X1, X2, X3, X4, X5))
        self.Y = np.hstack((y1, y2, y3, y4, y5))

        plt.scatter(self.X[:, 0], self.X[:, 1], c=self.Y)
        plt.show()

        # shuffle the data
        perm = range(0, self.n)
        shuffle(perm)
        self.X = self.X[perm]
        self.Y = self.Y[perm]

    def generate_human_prediction(self):
        self.human_pred = {}
        self.c = {}
        self.h = {}

        for std in self.list_of_std:
            h = np.zeros(self.Y.shape)
            self.human_pred[str(std)] = np.ones(shape=(self.Y.shape))
            for idx, label in enumerate(self.Y):

                if label == 1:
                    h[idx] = np.random.uniform(-0.2, 1)
                if label == -1:
                    h[idx] = np.random.uniform(-1, 0.2)
                if h[idx] * self.Y[idx] < 0:
                    self.human_pred[str(std)][idx] = -self.Y[idx]
                else:
                    self.human_pred[str(std)][idx] = self.Y[idx]
                self.c[str(std)] = np.maximum(0, 1 - (self.Y * h))
            self.h[str(std)] = h


    def split_data(self, frac):
        indices = np.arange(self.n)
        random.shuffle(indices)
        num_train = int(frac * self.n)
        indices_train = indices[:num_train]
        indices_test = indices[num_train:]
        self.Xtest = self.X[indices_test]
        self.Xtrain = self.X[indices_train]
        self.Ytrain = self.Y[indices_train]
        self.Ytest = self.Y[indices_test]
        self.human_pred_train = {}
        self.human_pred_test = {}
        self.c_train = {}
        self.c_test = {}

        for std in self.list_of_std:
            self.human_pred_train[str(std)] = self.human_pred[str(std)][indices_train]
            self.human_pred_test[str(std)] = self.human_pred[str(std)][indices_test]
            self.c_train[str(std)] = self.c[str(std)][indices_train]
            self.c_test[str(std)] = self.c[str(std)][indices_test]

        n_test = self.Xtest.shape[0]
        n_train = self.Xtrain.shape[0]
        self.dist_mat = np.zeros((n_test, n_train))

        for te in range(n_test):
            for tr in range(n_train):
                self.dist_mat[te, tr] = LA.norm(self.Xtest[te] - self.Xtrain[tr])


def convert(input_data, output_data):
    data = load_data(input_data, 'ifexists')
    list_of_std_str = data.human_pred_train.keys()
    test = {'X': data.Xtest, 'Y': data.Ytest, 'c': {}, 'y_h': {}}
    data_dict = {'test': test, 'X': data.Xtrain, 'Y': data.Ytrain, 'c': {}, 'y_h': {}, 'dist_mat': data.dist_mat}

    for std in list_of_std_str:
        data_dict['c'][std] = data.c_train[std]
        data_dict['test']['c'][std] = data.c_test[std]
        data_dict['y_h'][std] = data.human_pred_train[std]
        data_dict['test']['y_h'][std] = data.human_pred_test[std]

    save(data_dict, output_data)


def main():
    frac = 0.8
    list_of_options = ['Linear', 'Kernel']
    options = sys.argv[1:]

    if not os.path.exists('data'):
        os.mkdir('data')

    for option in options:
        assert option in list_of_options

        input_data_file = 'data/' + option

        if option == 'Linear':
            list_of_std = [1]
            obj = generate_data(n=320, dim=2, list_of_std=list_of_std, std_y=2)
            obj.c_gaussian_linear_generate_data()
            obj.generate_human_prediction()
            obj.split_data(frac)
            save(obj, input_data_file)

        if option == 'Kernel':
            list_of_std = [1]
            obj = generate_data(n=320, dim=2, list_of_std=list_of_std, std_y=2)
            obj.c_gaussian_kernel_generate_data()
            obj.generate_human_prediction()
            obj.split_data(frac)
            save(obj, input_data_file)

        if os.path.exists('data/data_dict_' + option + '.pkl'):
            os.remove('data/data_dict_' + option + '.pkl')
        output_data_file = 'data/data_dict_' + option
        print 'converting'
        convert(input_data_file, output_data_file)


if __name__ == "__main__":
    main()
