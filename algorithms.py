import math
from baseline_classes import *
import numpy as np
import random


class HumanMachine:
    def __init__(self, data_dict):
        self.X = data_dict['X']
        self.Y = data_dict['Y']
        self.c = data_dict['c']
        self.Pr_H = data_dict['Pr_H']
        self.Pr_M = data_dict['Pr_M']
        self.Pr_M_Alg = data_dict['Pr_M_Alg']
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)

    def get_Sc(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def get_added(self, subset, elm):
        return np.concatenate((subset, np.array([int(elm)])), axis=0)

    def set_param(self, lamb, K):
        self.lamb = lamb
        self.K = K

    def stochastic_distort_greedy(self, g, K, gamma, epsilon, svm_type):
        c_mod = C({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb, 'svm_type': svm_type})
        subset = np.array([]).astype(int)
        g.reset()
        s = int(math.ceil(self.n * np.log(float(1) / epsilon) / float(K)))

        print 'subset_size', s, 'K-->', K, ', n --> ', self.n

        for itr in range(K):
            frac = (1 - gamma / float(K)) ** (K - itr - 1)
            subset_c = self.get_Sc(subset)

            if s < subset_c.shape[0]:
                subset_chosen = np.array(random.sample(subset_c, s))
            else:
                subset_chosen = subset_c

            c_mod_inc = c_mod.get_inc_arr(subset, rest_flag=True, subset_rest=subset_chosen)
            g_inc_arr, subset_c_ret = g.get_inc_arr(subset, rest_flag=True, subset_rest=subset_chosen)
            g_pos_inc = g_inc_arr.flatten()

            inc_vec = frac * g_pos_inc - c_mod_inc

            if np.max(inc_vec) <= 0:
                print 'no increment'
                return subset

            sel_ind = np.argmax(inc_vec)
            elm = subset_chosen[sel_ind]
            subset = self.get_added(subset, elm)
            g.update_data_str(elm)

        return subset

    def distort_greedy(self, g, K, gamma, svm_type):
        c_mod = C({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb, 'svm_type': svm_type})
        subset = np.array([]).astype(int)
        g.reset()

        for itr in range(K):
            frac = (1 - gamma / float(K)) ** (K - itr - 1)
            subset_c = self.get_Sc(subset)
            c_mod_inc = c_mod.get_inc_arr(subset).flatten()
            g_inc_arr, subset_c_ret = g.get_inc_arr(subset)
            g_pos_inc = g_inc_arr.flatten()  # + c_mod_inc
            inc_vec = frac * g_pos_inc - c_mod_inc

            if np.max(inc_vec) <= 0:
                print 'no increment'
                return subset

            sel_ind = np.argmax(inc_vec)
            elm = subset_c[sel_ind]
            subset = self.get_added(subset, elm)
            g.update_data_str(elm)
        return subset

    def gamma_sweep_distort_greedy(self, svm_type, delta=0.001, T=1, flag_stochastic=None):

        g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb, 'svm_type': svm_type})
        subset = {}
        G_subset = []
        gamma = 0.01

        for t in range(T):
            if flag_stochastic:
                subset_sel = self.stochastic_distort_greedy(g, self.K, gamma, delta, svm_type=svm_type)
            else:
                subset_sel = self.distort_greedy(g, self.K, gamma, svm_type=svm_type)

            subset[str(t)] = subset_sel
            G_subset.append(g.eval(subset_sel))
            gamma = gamma * (1 - delta)

        max_set_ind = np.argmax(np.array(G_subset))
        return subset[str(max_set_ind)]

    def triage_subset(self, triage_type):
        triage_obj = Triage({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb,
                             'Pr_H': self.Pr_H, 'Pr_M': self.Pr_M, 'Pr_M_Alg': self.Pr_M_Alg})
        return triage_obj.get_subset(self.K, triage_type)

    def algorithms(self, param, optim):
        self.set_param(param['lamb'], int(param['K'] * self.n))

        if optim == 'distort_greedy':
            subset = self.gamma_sweep_distort_greedy(svm_type=param['svm_type'], T=1)

        if optim == 'stochastic_distort_greedy':
            subset = self.gamma_sweep_distort_greedy(flag_stochastic=True, T=3, svm_type=param['svm_type'])

        if optim == 'kl_triage_estimated':
            subset = self.triage_subset(triage_type='estimated')

        if optim == 'kl_triage_Alg':
            subset = self.triage_subset(triage_type='Alg')

        res = {'subset': subset}

        return res
