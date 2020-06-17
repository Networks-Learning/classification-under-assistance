from myutil import *
from algorithms import HumanMachine
import parser
import numpy as np


def parse_command_line_input(dataset):
    list_of_option = ['kl_triage_Alg', 'kl_triage_estimated', 'stochastic_distort_greedy', 'distort_greedy']
    list_of_real = ['Messidor', 'Aptos', 'Stare']
    list_of_synthetic = ['hard_linear', 'Linear', 'Kernel']
    list_of_std = [1]

    if dataset in list_of_real:
        list_of_K = [0.0, 0.04, 0.08, 0.12, 0.16, 0.2]
    if dataset in list_of_synthetic:
        list_of_K = [0.1, 0.4]

    assert (dataset in list_of_real or dataset in list_of_synthetic)

    if dataset == 'Messidor':
        threshold = 1.5
        list_of_lamb = [0.01]

    if dataset == 'Stare':
        threshold = 0.5
        list_of_lamb = [0.5]

    if dataset == 'Aptos':
        threshold = 1.8
        list_of_lamb = [0.6]

    if dataset in ['hard_linear', 'Linear', 'Kernel']:
        threshold = 0
        list_of_lamb = [1]
        list_of_option = ['stochastic_distort_greedy', 'distort_greedy']

    return list_of_K, list_of_option, list_of_std, list_of_lamb, threshold


class Eval:

    def __init__(self, data_file, list_of_K, list_of_std, list_of_lamb, list_of_option, threshold=None, real_flag=None,
                 real_wt_std=None):
        self.data = load_data(data_file)
        self.real = real_flag
        self.real_wt_std = real_wt_std
        self.list_of_K = list_of_K
        self.list_of_std = list_of_std
        self.list_of_lamb = list_of_lamb
        self.list_of_option = list_of_option
        self.threshold = threshold
        self.list_of_real = ['Messidor', 'Aptos', 'Stare']
        self.list_of_synthetic = ['hard_linear', 'Linear', 'Kernel']

    def get_labels(self, cont_y):
        y = np.zeros(cont_y.shape)

        for idx, label in enumerate(cont_y):
            if label > self.threshold:
                y[idx] = 1
            else:
                y[idx] = -1
        return y

    def compute_triage_alg(self, X_tr, Y_tr, X_te, lamb):

        from sklearn.svm import SVC
        reg_par = float(1) / (2.0 * lamb * X_tr.shape[0])
        machine_model = SVC(C=reg_par, kernel='linear')
        machine_model.fit(X_tr, Y_tr)

        h = np.absolute(machine_model.decision_function(X_tr))
        h_test = np.absolute(machine_model.decision_function(X_te))
        Pr_M = 1.0 / (1.0 + np.exp(h))
        Pr_M_test = 1.0 / (1.0 + np.exp(h_test))

        return Pr_M, Pr_M_test

    def compute_kl_triage_estimated(self, X_tr, Y_tr, X_te, y_h, Pr_H, lamb):
        from sklearn.neural_network import MLPClassifier
        model = MLPClassifier()
        model.fit(X_tr, Pr_H)
        Pr_H_test = model.predict(X_te)

        from sklearn.svm import SVC
        reg_par = float(1) / (2.0 * lamb * X_tr.shape[0])
        machine_model = SVC(C=reg_par, kernel='linear')
        machine_model.fit(X_tr, Y_tr)
        train_pred = machine_model.predict(X_tr)
        Y = np.zeros(X_tr.shape[0], dtype='int')
        for idx, item in enumerate(y_h):
            if item != train_pred[idx]:  # disagreement between human and machine
                Y[idx] = 1

        model = MLPClassifier()
        model.fit(X_tr, Y)
        Pr_M_test = model.predict(X_te)
        return Pr_H_test, Y, Pr_M_test

    def prepare_for_train(self, res_file, file_name):
        res = load_data(res_file, 'ifexists')
        split = 3
        frac = int(self.data['X'].shape[0] / 10) * 4  # test split size

        X_tr = self.data['X'][frac:]
        Y_tr = self.get_labels(self.data['Y'][frac:])

        c = self.data['c'][str(self.list_of_std[0])][frac:]  # slack human error
        c_te = self.data['c'][str(self.list_of_std[0])][:frac]

        X_te = self.data['X'][:frac]
        Y_te = self.get_labels(self.data['Y'][:frac])

        y_h = self.data['y_h'][str(self.list_of_std[0])][frac:]  # human prediction
        y_h_test = self.data['y_h'][str(self.list_of_std[0])][:frac]

        if file_name in self.list_of_real:
            Pr_H = self.data['Pr_H'][str(self.list_of_std[0])][frac:]

            Pr_M_Alg, Pr_M_Alg_test = self.compute_triage_alg(X_tr, Y_tr, X_te,
                                                              self.list_of_lamb[0])  # Algorithmic triage subsets

            Pr_H_test, Pr_M, Pr_M_test = self.compute_kl_triage_estimated(X_tr, Y_tr, X_te, y_h, Pr_H,
                                                                          self.list_of_lamb[0])
        if file_name in self.list_of_synthetic:
            Pr_H = Pr_M = Pr_M_Alg = np.zeros(3 * frac)
            Pr_H_test = Pr_M_test = Pr_M_Alg_test = np.zeros(frac)

        if str(split) not in res:
            res[str(split)] = {}

        res[str(split)]['X_tr'] = X_tr
        res[str(split)]['Y_tr'] = Y_tr
        res[str(split)]['X_te'] = X_te
        res[str(split)]['Y_te'] = Y_te
        res[str(split)]['Pr_M_Alg_test'] = Pr_M_Alg_test
        res[str(split)]['Pr_M_Alg'] = Pr_M_Alg
        res[str(split)]['Pr_M_test'] = Pr_M_test
        res[str(split)]['Pr_H_test'] = Pr_H_test
        res[str(split)]['Pr_H'] = Pr_H
        res[str(split)]['Pr_M'] = Pr_M
        res[str(split)]['y_h'] = y_h
        res[str(split)]['y_h_test'] = y_h_test
        res[str(split)]['c'] = c
        res[str(split)]['c_te'] = c_te
        save(res, res_file)

    def train(self, res_file, svm_type):

        res = load_data(res_file, 'ifexists')
        split = 3
        X_tr = res[str(split)]['X_tr']
        Y_tr = res[str(split)]['Y_tr']
        Pr_H = res[str(split)]['Pr_H']
        Pr_M = res[str(split)]['Pr_M']
        Pr_M_Alg = res[str(split)]['Pr_M_Alg']
        c = res[str(split)]['c']

        for option in self.list_of_option:
            for std in self.list_of_std:
                for i, K in enumerate(self.list_of_K):
                    for lamb in self.list_of_lamb:
                        if str(std) not in res[str(split)]:
                            res[str(split)][str(std)] = {}
                        if str(K) not in res[str(split)][str(std)]:
                            res[str(split)][str(std)][str(K)] = {}
                        if str(lamb) not in res[str(split)][str(std)][str(K)]:
                            res[str(split)][str(std)][str(K)][str(lamb)] = {}

                        local_data = {'X': X_tr, 'Y': Y_tr, 'c': c,
                                      'Pr_H': Pr_H, 'Pr_M': Pr_M,
                                      'Pr_M_Alg': Pr_M_Alg}

                        triage_obj = HumanMachine(local_data)

                        if K != 0:
                            res_dict = triage_obj.algorithms({'K': K, 'lamb': lamb, 'svm_type': svm_type},
                                                             optim=option)
                        else:
                            res_dict = {'subset': np.array([])}

                        res_dict['subset_test'] = {}
                        res[str(split)][str(std)][str(K)][str(lamb)][option] = res_dict
                        save(res, res_file)

    def get_test_subset(self, res_file, K, subset, subset_prev):
        split = 3
        res = load_data(res_file)
        x_tr = res[str(split)]['X_tr']
        x_te = res[str(split)]['X_te']
        y_tr = res[str(split)]['Y_tr']
        Pr_M_Alg_test = res[str(split)]['Pr_M_Alg_test']
        Pr_H = res[str(split)]['Pr_H_test']
        Pr_M = res[str(split)]['Pr_M_test']

        subset_c = np.array([int(i) for i in range(x_tr.shape[0]) if i not in subset_prev])

        from sklearn.neural_network import MLPClassifier
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC

        n = x_te.shape[0]
        tr_n = x_tr.shape[0]
        no_human = int((subset.shape[0] * n) / float(tr_n))

        if K == 0:
            subset_te_r = np.array([])
            return subset_te_r, subset_te_r, subset_te_r, subset_te_r

        reg_par = float(1) / (2.0 * self.list_of_lamb[0] * subset_c.shape[0])
        svm_model = SVC(kernel='linear', C=reg_par)
        svm_model.fit(x_tr[subset_c], y_tr[subset_c])

        features = np.absolute(svm_model.decision_function(x_tr[subset_c]))
        features = np.expand_dims(features, 1)

        features_test = np.absolute(svm_model.decision_function(x_te))
        features_test = np.expand_dims(features_test, 1)

        y = np.zeros(x_tr.shape[0], dtype='uint')
        y[subset] = 1  # human label = 1

        modelMLP = MLPClassifier(solver='lbfgs')
        modelLR = LogisticRegression(C=10)

        modelMLP.fit(x_tr[subset_c], y[subset_c])
        modelLR.fit(features, y[subset_c])

        y_predMLP = np.argsort(modelMLP.predict_proba(x_te)[:, 1])[-no_human:]

        y_predLR = np.argsort(modelLR.predict_proba(features_test)[:, 1])[-no_human:]

        subset_te_MLP = y_predMLP
        subset_te_LR = y_predLR

        err = - Pr_M_Alg_test
        subset_te_Alg = np.argsort(err)[:no_human]

        err = Pr_H - Pr_M
        subset_te_Est = np.argsort(err)[:no_human]

        return subset_te_MLP, subset_te_LR, subset_te_Alg, subset_te_Est

    def test_subset(self, res_file, option):
        res = load_data(res_file)
        split = 3
        for std in self.list_of_std:
            for i, K in enumerate(self.list_of_K):
                for lamb in self.list_of_lamb:
                    res_dict = res[str(split)][str(std)][str(K)][str(lamb)][option]
                    subset = res_dict['subset']
                    if K != 0:
                        subset_prev = res[str(split)][str(std)][str(self.list_of_K[i - 1])][str(lamb)][option]['subset']
                    else:
                        subset_prev = subset

                    subset_te_MLP, subset_te_LR, subset_te_Alg, subset_te_Est = self.get_test_subset(res_file=res_file,
                                                                                                     K=K,
                                                                                                     subset=subset,
                                                                                                     subset_prev=subset_prev)
                    res[str(split)][str(std)][str(K)][str(lamb)][option] = {'subset': subset,
                                                                            'subset_test': {'MLP': subset_te_MLP,
                                                                                            'LR': subset_te_LR,
                                                                                            'Alg': subset_te_Alg,
                                                                                            'Est': subset_te_Est}}

        save(res, res_file)


def main():
    my_parser = parser.opts_parser()
    args = my_parser.parse_args()
    args = vars(args)
    file_name = [args['dataset']][0]
    svm_type = args['svm_type']

    print 'training ' + file_name
    list_of_K, list_of_option, list_of_std, list_of_lamb, threshold = parse_command_line_input(file_name)
    data_file = 'data/data_dict_' + file_name

    if not os.path.exists('Results'):
        os.mkdir('Results')

    res_file = 'Results/' + file_name + '_' + svm_type + '_res_' + str(list_of_lamb[0])

    # if os.path.exists(res_file + '.pkl'):
    #     os.remove(res_file + '.pkl')

    obj = Eval(data_file, list_of_K, list_of_std, list_of_lamb, list_of_option, threshold)
    obj.prepare_for_train(res_file=res_file, file_name=file_name)
    obj.train(res_file=res_file, svm_type=svm_type)
    if file_name in obj.list_of_real:
        for option in list_of_option:
            obj.test_subset(res_file, option)


if __name__ == "__main__":
    main()
