from myutil import *
import numpy as np
import matplotlib.pyplot as plt
from plot_util import latexify
import parser
from sklearn import svm


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
        list_of_lamb = [0.03]  # 0.03

    if dataset == 'Stare':
        threshold = 0.5
        list_of_lamb = [0.5]

    if dataset == 'Aptos':
        threshold = 1.8
        list_of_lamb = [0.6]

    if dataset in ['hard_linear', 'Linear', 'Kernel']:
        threshold = 0
        list_of_lamb = [1]

    return list_of_K, list_of_option, list_of_std, list_of_lamb, threshold


class plot_triage_real:

    def __init__(self, list_of_K, list_of_std, list_of_lamb, list_of_option, threshold=0, flag_synthetic=None):
        self.list_of_K = list_of_K
        self.list_of_std = list_of_std
        self.list_of_lamb = list_of_lamb
        self.list_of_option = list_of_option
        self.flag_synthetic = flag_synthetic
        self.threshold = threshold
        self.test_map = {'stochastic_distort_greedy': 'LR', 'distort_greedy': 'LR',
                         'kl_triage_estimated': 'Est', 'kl_triage_Alg': 'Alg'}

    def plot_subset(self, res_file, path, svm_type, option):
        res = load_data(res_file)
        split = 3
        X_tr = res[str(split)]['X_tr']
        Y_tr = res[str(split)]['Y_tr']
        X_te = res[str(split)]['X_te']
        Y_te = res[str(split)]['Y_te']
        c = res[str(split)]['c']

        lamb = self.list_of_lamb[0]
        for K in self.list_of_K:

            for std in self.list_of_std:
                fig, ax = plt.subplots()
                fig.subplots_adjust(left=.15, bottom=.16, right=.99, top=0.93)
                local_data = {'X': X_tr, 'Y': Y_tr, 'c': c,
                              'X_te': X_te, 'Y_te': Y_te}
                local_res = res[str(split)][str(std)][str(K)][str(lamb)][option]
                subset_human = local_res['subset']
                n = local_data['X'].shape[0]
                subset_machine = np.array([i for i in range(n) if i not in subset_human])

                machine_plus = np.array([idx for idx in subset_machine if Y_tr[idx] > 0])
                machine_minus = np.array([idx for idx in subset_machine if Y_tr[idx] <= 0])
                human_plus = np.array([idx for idx in subset_human if Y_tr[idx] > 0])
                human_minus = np.array([idx for idx in subset_human if Y_tr[idx] <= 0])

                X_machine = X_tr[subset_machine]
                Y_machine = Y_tr[subset_machine]

                reg_par = float(1) / (2.0 * lamb * subset_machine.shape[0])

                if svm_type == 'hard_Linear':
                    model = svm.LinearSVC(C=1000, loss='hinge')

                if svm_type == 'hard_linear_without_offset':
                    model = svm.LinearSVC(fit_intercept=False, C=1000, loss='hinge')

                if svm_type == 'soft_linear_with_offset':
                    model = svm.SVC(kernel='linear', C=reg_par)

                if svm_type == 'soft_linear_without_offset':
                    model = svm.LinearSVC(fit_intercept=False, C=reg_par, loss='hinge')

                if svm_type == 'soft_kernel_with_offset':
                    model = svm.SVC(kernel='poly', degree=2, C=reg_par, gamma='auto')

                model.fit(X_machine, Y_machine)

                plt.scatter(X_tr[machine_plus, 0], X_tr[machine_plus, 1], color='darkcyan', marker='o', lw=1.5,
                            label=r'$\mathcal{V}$\textbackslash$\mathcal{S}, y=1$', facecolor='none', s=60, zorder=30)
                plt.scatter(X_tr[machine_minus, 0], X_tr[machine_minus, 1], marker='o', lw=1.5, s=60, facecolor='none',
                            label=r'$\mathcal{V}$\textbackslash $\mathcal{S}, y=-1$', color='salmon', zorder=30)

                if human_plus.shape[0] > 0:
                    plt.scatter(X_tr[human_plus, 0], X_tr[human_plus, 1], marker='o', s=60, linewidth=2,
                                color='darkcyan', label=r'$\mathcal{S},y=1$', zorder=30)

                if human_minus.shape[0] > 0:
                    plt.scatter(X_tr[human_minus, 0], X_tr[human_minus, 1], marker='o', s=60, linewidth=2,
                                color='salmon', label=r'$\mathcal{S},y=-1$', zorder=30)

                x_min = -12
                x_max = 12
                y_min = -12
                y_max = 12

                XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j]
                Z = model.decision_function(np.c_[XX.ravel(), YY.ravel()])

                Z = Z.reshape(XX.shape)
                plt.contour(XX, YY, Z, colors=['salmon', 'dimgrey', 'darkcyan'], linewidths=2,
                            linestyles=['--', '-', '--'],
                            levels=[-1, 0, 1])

                ax.tick_params(direction='out', length=6, width=2, labelsize=28,
                               grid_alpha=1)

                ax.set_xlim(-12, 12)
                ax.set_ylim(-12, 12)
                option_map = {'stochastic_distort_greedy': 'sdg', 'distort_greedy': 'dg', 'kl_triage_Alg': 'Alg',
                              'kl_triage_estimated': 'Est'}

                savepath = path + svm_type + '_' + str(K) + '_' + option_map[option]
                plt.savefig(savepath + '.pdf')
                plt.savefig(savepath + '.png')
                plt.close()

    def get_mean_vary_K(self, res_file, image_path, file_name):
        res = load_data(res_file)
        split = 3
        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}
                for option in self.list_of_option:
                    err_K_te = []
                    for K in self.list_of_K:
                        err_K_te.append(res[str(split)][str(std)][str(K)][str(lamb)][option]['test_res']
                                        [self.test_map[option]][self.test_map[option]]['error'])

                    plot_obj[option] = {'test': err_K_te}

                self.plot_mean_vs_K(res=res, image_file=image_path, plot_obj=plot_obj, file_name=file_name,
                                    std=std, lamb=lamb)

    def get_train_test_error_vary_K(self, res_file, image_path, file_name):
        res = load_data(res_file)
        split = 3
        X_tr = res[str(split)]['X_tr']
        Y_tr = res[str(split)]['Y_tr']
        y_h = res[str(split)]['y_h']
        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}

                for option in ['distort_greedy']:
                    err_K_tr = []
                    err_K_te = []
                    for K in self.list_of_K:
                        err_K_tr.append(
                            res[str(split)][str(std)][str(K)][str(lamb)][option]['train_res']['error'])
                        err_K_te.append(
                            res[str(split)][str(std)][str(K)][str(lamb)][option]['test_res'][self.test_map[option]][
                                self.test_map[option]]['error'])

                    plot_obj[option] = {'train': err_K_tr, 'test': err_K_te}

                self.train_plot_err_vs_K(image_path, plot_obj, file_name, std, lamb, x_tr=X_tr, split=split,
                                         y_tr=Y_tr, y_h=y_h)

    def plot_f1(self, res_file, image_path, file_name):
        res = load_data(res_file)

        for std in self.list_of_std:
            for lamb in self.list_of_lamb:
                plot_obj = {}
                for option in self.list_of_option:
                    err_K_te = []
                    for K in self.list_of_K:
                        err_K_te.append(
                            res['3'][str(std)][str(K)][str(lamb)][option]['test_res'][self.test_map[option]]['f_score'])

                    plot_obj[option] = {'test': err_K_te}

                self.plot_f1_score(res=res, image_file=image_path, plot_obj=plot_obj, file_name=file_name,
                                   std=std, lamb=lamb)

    def plot_mean_vs_K(self, res, image_file, plot_obj, file_name, std, lamb):

        savepath = image_file + file_name
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.21, bottom=.20, right=.99, top=.9)
        color_list = get_color_list()

        human_error = []
        machine_error = []
        for split in range(3, 4):
            X_tr = res[str(split)]['X_tr']
            Y_tr = res[str(split)]['Y_tr']
            X_te = res[str(split)]['X_te']
            Y_te = res[str(split)]['Y_te']
            y_h_test = res[str(split)]['y_h_test']
            human_pred = y_h_test
            machine_pred = self.get_machine_pred(X_tr, Y_tr, X_te)
            true_pred = Y_te
            human_error.append(np.sum(human_pred != true_pred) / float(true_pred.shape[0]))
            machine_error.append(np.sum(machine_pred != true_pred) / float(true_pred.shape[0]))

        human_obj = []
        machine_obj = []
        key = 'test'
        for idx, option in enumerate(plot_obj.keys()):
            err = [x for x in plot_obj[option][key]]
            label_map = {'kl_triage_estimated': 'Estimated Triage',
                         'kl_triage_Alg': 'Alg Triage',
                         'distort_greedy': 'DG',
                         'stochastic_distort_greedy': 'Stochastic DG'}

            ax.plot(err, label=label_map[option], linewidth=4, marker='o',
                    markersize=12, color=color_list[idx])

        human_obj.append(human_error)
        machine_obj.append(machine_error)
        plt.scatter([0.0], machine_obj, marker='^', s=450, zorder=30, color='red')
        ax.legend()

        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=32)
        if file_name == 'messidor' or file_name == 'Messidor':
            ax.set_ylabel(r'$\mathbb{P}(y\neq\hat{y})$', fontsize=36, labelpad=5)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        if file_name == 'Aptos':
            plt.yticks([0.144, 0.175])
            plt.yticks([0.15, 0.16, 0.17])

        if file_name == 'Stare':
            plt.ylim(0.179, 0.243)
            plt.yticks([0.19, 0.21, 0.23])

        if file_name == 'Messidor':
            plt.ylim(0.254, 0.348)
            plt.yticks([0.27, 0.3, 0.33])

        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '.pdf')
        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '.png')
        plt.close()

    def get_machine_pred(self, X_tr, Y_tr, X_te):
        from sklearn.svm import SVC
        lamb = self.list_of_lamb[0]
        reg_par = float(1) / (2.0 * lamb * X_tr.shape[0])
        model = SVC(kernel='linear', C=reg_par)
        model.fit(X_tr, Y_tr)
        y_pred = model.predict(X_te)
        return y_pred

    def train_plot_err_vs_K(self, image_file, plot_obj, file_name, std, lamb, x_tr, y_tr, y_h, split):

        savepath = image_file + file_name + '_train'
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.22, bottom=.20, right=.99, top=.9)
        color_list = get_color_list()

        for idx, option in enumerate(plot_obj):
            for i, key in enumerate(plot_obj[option]):
                err = [x for x in plot_obj[option][key]]
                label_map = {'kl_triage_estimated': 'Estimated Triage',
                             'kl_triage_Alg': 'Alg Triage',
                             'distort_greedy': 'DG',
                             'stochastic_distort_greedy': 'Stochastic DG'}
                ax.plot(err, linewidth=3, marker='o', color=color_list[i],
                        markersize=12, label=key + '-' + label_map[option])

        human_pred = y_h
        machine_pred = self.get_machine_pred(x_tr, y_tr, x_tr)

        true_pred = y_tr
        human_error = np.sum(human_pred != true_pred) / float(true_pred.shape[0])
        machine_error = np.sum(machine_pred != true_pred) / float(true_pred.shape[0])
        human_obj = []
        machine_obj = []
        for K in self.list_of_K:
            human_obj.append(human_error)
            machine_obj.append(machine_error)

        handles, labels = plt.gca().get_legend_handles_labels()
        order = [0, 1]
        ax.legend([handles[idx] for idx in order], ['Test set', 'Train set'], prop={'size': 16}, frameon=False,
                  handlelength=1, handletextpad=0.4)

        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=32)
        ax.set_ylabel(r'$\mathbb{P}(y\neq\hat{y})$', fontsize=34, labelpad=5)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '_' + str(split) + '.pdf')
        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '_' + str(split) + '.png')
        plt.close()

    def plot_f1_score(self, res, image_file, plot_obj, file_name, std, lamb):

        savepath = image_file + file_name + '_f'
        fig, ax = plt.subplots()
        fig.subplots_adjust(left=.21, bottom=.20, right=.99, top=.9)
        color_list = get_color_list()
        human_f1 = []
        machine_f1 = []
        for split in range(3, 4):
            X_tr = res[str(split)]['X_tr']
            Y_tr = res[str(split)]['Y_tr']
            X_te = res[str(split)]['X_te']
            Y_te = res[str(split)]['Y_te']
            y_h_test = res[str(split)]['y_h_test']
            human_pred = y_h_test
            machine_pred = self.get_machine_pred(X_tr, Y_tr, X_te)

            from sklearn.metrics import f1_score
            human_f1.append(f1_score(Y_te, human_pred))
            machine_f1.append(f1_score(Y_te, machine_pred))
        key = 'test'

        for idx, option in enumerate(plot_obj.keys()):
            err = [x for x in plot_obj[option][key]]
            label_map = {'kl_triage_estimated': 'Estimated Triage',
                         'kl_triage_Alg': 'Alg Triage',
                         'distort_greedy': 'DG',
                         'stochastic_distort_greedy': 'Stochastic DG'}

            ax.plot(err, label=label_map[option], linewidth=4, marker='o',
                    markersize=12, color=color_list[idx])
        plt.scatter([0.0], machine_f1, marker='^', zorder=30, s=450, color='red')

        plt.xlabel(r'$n/ | \mathcal{V} | $', fontsize=32)
        if file_name == 'messidor' or file_name == 'Messidor':
            ax.set_ylabel(r'F1 Score', fontsize=36, labelpad=5)
        plt.xticks(range(len(self.list_of_K)), self.list_of_K)

        if file_name == 'Aptos':
            plt.ylim([0.825, 0.855])
            plt.yticks([0.83, 0.84, 0.85])

        if file_name == 'Stare':
            plt.ylim(0.635, 0.727)
            plt.yticks([0.65, 0.68, 0.71])

        if file_name == 'Messidor':
            plt.ylim([0.644, 0.76])
            plt.yticks([0.66, 0.7, 0.74])

        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '.pdf')
        plt.savefig(savepath + '_' + str(std) + '_' + str(lamb) + '.png')
        plt.close()

    def classification_get_test_error(self, model, res_obj, x_te, y_te,
                                      option, y_h_test=None):

        subset_test = res_obj['subset_test'][self.test_map[option]]
        subset_test_c = np.array([int(i) for i in range(x_te.shape[0]) if i not in subset_test])

        y_pred = model.predict(x_te[subset_test_c])
        err_m = (y_pred != y_te[subset_test_c])

        from sklearn.metrics import f1_score
        if subset_test.size == 0:
            final_y_pred = y_pred
            final_y_true = y_te
        else:
            final_y_pred = np.concatenate((y_pred, y_h_test[subset_test]))
            final_y_true = np.concatenate((y_te[subset_test_c], y_te[subset_test]))
        f_score = f1_score(final_y_true, final_y_pred)

        if subset_test.size == 0:
            error_r = float(err_m.sum()) / float(x_te.shape[0])
        else:
            err_h = (y_h_test[subset_test] != y_te[subset_test])
            error_r = (err_h.sum() + err_m.sum()) / float(x_te.shape[0])

        error_n = {'error': error_r, 'human_ind': subset_test, 'machine_ind': subset_test_c}
        error_r = {'error': error_r, 'human_ind': subset_test, 'machine_ind': subset_test_c}

        return error_n, error_r, f_score

    def get_train_error(self, res_obj, x, y, subset_prev, option, y_h=None):
        subset = res_obj['subset']

        if subset.shape[0] != 0:
            err_h = (y[subset] != y_h[subset])
        else:
            err_h = np.zeros(subset)

        subset_c = np.array([int(i) for i in range(x.shape[0]) if i not in subset_prev])
        subset_c_new = np.array([int(i) for i in range(x.shape[0]) if i not in subset])
        from sklearn import svm

        if option in ['stochastic_distort_greedy', 'distort_greedy']:
            reg_par = float(1) / (2.0 * self.list_of_lamb[0] * subset_c.shape[0])
            model = svm.SVC(kernel='linear', C=reg_par)
            model.fit(x[subset_c], y[subset_c])

        if option in ['kl_triage_Alg', 'kl_triage_estimated']:
            reg_par = float(1) / (2.0 * self.list_of_lamb[0] * x.shape[0])
            model = svm.SVC(kernel='linear', C=reg_par)
            model.fit(x, y)

        y_pred = model.predict(x[subset_c_new])
        err_m = (y_pred != y[subset_c_new])

        error = (err_h.sum() + err_m.sum()) / float(x.shape[0])
        return {'error': error}, model

    def get_labels(self, cont_y):
        y = np.zeros(cont_y.shape)
        for idx, label in enumerate(cont_y):
            if label > self.threshold:
                y[idx] = 1
            else:
                y[idx] = -1
        return y

    def compute_result(self, res_file, option):
        res = load_data(res_file)
        split = 3
        X_tr = res[str(split)]['X_tr']
        Y_tr = res[str(split)]['Y_tr']
        X_te = res[str(split)]['X_te']
        Y_te = res[str(split)]['Y_te']
        y_h = res[str(split)]['y_h']
        y_h_test = res[str(split)]['y_h_test']

        for std in self.list_of_std:
            for i, K in enumerate(self.list_of_K):
                for lamb in self.list_of_lamb:
                    if option in res[str(split)][str(std)][str(K)][str(lamb)]:
                        res_obj = res[str(split)][str(std)][str(K)][str(lamb)][option]
                        if i != 0:
                            subset_prev = res[str(split)][str(std)][str(self.list_of_K[i - 1])][str(lamb)][option][
                                'subset']
                        else:
                            subset_prev = res[str(split)][str(std)][str(self.list_of_K[i])][str(lamb)][option]['subset']
                        train_res, model = self.get_train_error(res_obj, subset_prev=subset_prev, x=X_tr, y=Y_tr,
                                                                option=option,
                                                                y_h=y_h)

                        test_res_n, test_res_r, f_score = self.classification_get_test_error(model=model,
                                                                                             res_obj=res_obj,
                                                                                             x_te=X_te, y_te=Y_te,
                                                                                             y_h_test=y_h_test,
                                                                                             option=option)

                        if 'test_res' not in res[str(split)][str(std)][str(K)][str(lamb)][option]:
                            res[str(split)][str(std)][str(K)][str(lamb)][option]['test_res'] = {}

                        res[str(split)][str(std)][str(K)][str(lamb)][option]['test_res'][self.test_map[option]] = {
                            'ranking': test_res_r,
                            self.test_map[option]: test_res_n,
                            'f_score': f_score}
                        res[str(split)][str(std)][str(K)][str(lamb)][option]['train_res'] = train_res

        save(res, res_file)


def main():
    latexify()
    my_parser = parser.opts_parser()
    args = my_parser.parse_args()
    args = vars(args)
    list_of_file_names = [args['dataset']]
    svm_type = args['svm_type']

    image_path = 'plots/'
    if not os.path.exists(image_path):
        os.mkdir(image_path)

    for file_name in list_of_file_names:
        print 'plotting ' + file_name

        list_of_K, list_of_option, list_of_std, list_of_lamb, threshold = parse_command_line_input(file_name)
        res_file = 'Results/' + file_name + '_' + svm_type + '_res_' + str(list_of_lamb[0])

        obj = plot_triage_real(list_of_K, list_of_std, list_of_lamb, list_of_option, threshold=threshold)

        savepath = image_path + '/' + file_name + '/'

        if not os.path.exists(savepath):
            os.mkdir(savepath)

        if file_name in ['Kernel', 'Linear']:
            for option in ['stochastic_distort_greedy', 'distort_greedy']:
                obj.plot_subset(res_file=res_file,
                                path=savepath, svm_type=svm_type,
                                option=option)

        if file_name in ['Messidor', 'Aptos', 'Stare']:
            for option in list_of_option:
                obj.compute_result(res_file, option)

            obj.get_mean_vary_K(res_file, savepath,
                                file_name)
            obj.plot_f1(res_file, savepath,
                        file_name)
            # obj.get_train_test_error_vary_K(res_file, savepath, file_name)


if __name__ == "__main__":
    main()
