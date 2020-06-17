from myutil import *
from sklearn import svm
from sklearn.metrics import hinge_loss


class Triage:
    def __init__(self, data):
        self.X = data['X']
        self.Y = data['Y']
        self.Pr_H = data['Pr_H']
        self.Pr_M = data['Pr_M']
        self.Pr_M_Alg = data['Pr_M_Alg']
        self.n, self.dim = self.X.shape
        self.V = np.arange(self.n)

    def get_subset(self, K, triage_type):

        if triage_type == 'estimated':
            err = self.Pr_H - self.Pr_M

        if triage_type == 'Alg':
            err = - self.Pr_M_Alg

        indices = np.argsort(err)
        return indices[:K]


class C:
    def __init__(self, data):
        self.X = data['X']
        self.Y = data['Y']
        self.c = data['c']
        self.lamb = data['lamb']
        self.svm_type = data['svm_type']
        self.g = G({'X': self.X, 'Y': self.Y, 'c': self.c, 'lamb': self.lamb, 'svm_type': self.svm_type})
        self.n, self.dim = self.X.shape
        self.V = np.arange(self.n)

    def get_Sc(self, subset):
        return np.array([int(i) for i in range(self.n) if i not in subset])

    def get_inc_arr(self, subset, rest_flag=False, subset_rest=None):
        if rest_flag:
            subset_c = subset_rest
        else:
            subset_c = self.get_Sc(subset)

        return np.array([self.c[i] for i in subset_c]).flatten()


class G:
    def __init__(self, input):
        self.X = input['X']
        self.Y = input['Y']
        self.lamb = input['lamb']
        self.c = input['c']
        self.svm_type = input['svm_type']
        self.dim = self.X.shape[1]
        self.n = self.X.shape[0]
        self.V = np.arange(self.n)
        self.init_data_str()
        self.bigVal = 1000

    def reset(self):
        self.init_data_str()

    def get_Sc(self, subset):
        return np.array([int(i) for i in self.V if i not in subset])

    def init_data_str(self):
        self.c_S = 0
        self.curr_set_len = 0

    def get_hard_linear_svm_w_b(self, subset_c):
        x = self.X[subset_c]
        y = self.Y[subset_c]
        model = svm.LinearSVC(C=1000, loss='hinge')
        model.fit(x, y)
        w = model.coef_
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        return reg

    def get_hard_linear_svm_w(self, subset_c):
        x = self.X[subset_c]
        y = self.Y[subset_c]
        model = svm.LinearSVC(fit_intercept=False, C=1000, loss='hinge')
        model.fit(x, y)
        w = model.coef_
        b = model.intercept_
        assert (b == 0)
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        return reg

    def get_soft_linear_svm_w_b(self, subset_c):
        x = self.X[subset_c]
        y = self.Y[subset_c]
        reg_par = float(1) / (2.0 * self.lamb * subset_c.shape[0])
        model = svm.SVC(kernel='linear', C=reg_par)
        model.fit(x, y)
        y_pred = model.decision_function(x)
        w = model.coef_
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        hinge_machine_loss = hinge_loss(y, y_pred)
        hinge_machine_loss *= y_pred.shape[0]
        return reg + hinge_machine_loss

    def get_soft_linear_svm_w(self, subset_c):
        x = self.X[subset_c]
        y = self.Y[subset_c]

        reg_par = float(1) / (2.0 * self.lamb * subset_c.shape[0])
        model = svm.LinearSVC(fit_intercept=False, C=reg_par, loss='hinge')
        model.fit(x, y)
        y_pred = model.decision_function(x)

        w = model.coef_
        b = model.intercept_
        assert (b == 0)
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]

        hinge_machine_loss = hinge_loss(y, y_pred)
        hinge_machine_loss *= y_pred.shape[0]
        return reg + hinge_machine_loss

    def get_soft_kernel_svm_w_b(self, subset_c):
        x = self.X[subset_c]
        y = self.Y[subset_c]
        reg_par = float(1) / (2.0 * self.lamb * subset_c.shape[0])
        model = svm.SVC(C=reg_par, kernel='poly', degree=2, gamma='auto')
        model.fit(x, y)
        coef = model.dual_coef_
        sv = model.support_vectors_
        w = np.dot(coef, sv)
        reg = self.lamb * (subset_c.shape[0]) * np.dot(w, w.T)[0][0]
        y_pred = model.decision_function(x)
        hinge_machine_loss = hinge_loss(y, y_pred)
        hinge_machine_loss *= y_pred.shape[0]

        return reg + hinge_machine_loss

    def update_data_str(self, elm):
        self.c_S += self.c[elm]
        self.curr_set_len += 1

    def give_inc(self, subset, elm):
        subset = np.append(subset, np.array([elm]).astype(int))
        subset_c = self.get_Sc(subset)
        if self.svm_type == 'hard_linear_with_offset':
            machine_error = self.get_hard_linear_svm_w_b(subset_c)

        if self.svm_type == 'hard_linear_without_offset':
            machine_error = self.get_hard_linear_svm_w(subset_c)

        if self.svm_type == 'soft_linear_with_offset':
            machine_error = self.get_soft_linear_svm_w_b(subset_c)

        if self.svm_type == 'soft_linear_with_offset':
            machine_error = self.get_soft_linear_svm_w_b(subset_c)

        if self.svm_type == 'soft_kernel_with_offset':
            machine_error = self.get_soft_kernel_svm_w_b(subset_c)

        return - machine_error

    def eval_curr(self, subset_c):
        if self.svm_type == 'hard_linear_with_offset':
            machine_error = self.get_hard_linear_svm_w_b(subset_c)

        if self.svm_type == 'hard_linear_without_offset':
            machine_error = self.get_hard_linear_svm_w(subset_c)

        if self.svm_type == 'soft_linear_with_offset':
            machine_error = self.get_soft_linear_svm_w_b(subset_c)

        if self.svm_type == 'soft_linear_without_offset':
            machine_error = self.get_soft_linear_svm_w(subset_c)

        if self.svm_type == 'soft_kernel_with_offset':
            machine_error = self.get_soft_kernel_svm_w_b(subset_c)

        return - machine_error

    def get_inc_arr(self, subset, rest_flag=False, subset_rest=None):
        subset_c = self.get_Sc(subset)
        F_S = self.eval_curr(subset_c)

        if rest_flag:
            subset_c = subset_rest
        else:
            subset_c = self.get_Sc(subset)

        vec = []
        for i in subset_c:
            vec.append(self.give_inc(subset, i) - F_S)

        return np.array(vec), subset_c

    def eval(self, subset=None):

        subset_c = self.get_Sc(subset)

        if subset.size == 0:
            c_S = 0
        else:
            c_S = self.c[subset].sum()

        if self.svm_type == 'hard_linear_with_offset':
            machine_error = self.get_hard_linear_svm_w_b(subset_c)

        if self.svm_type == 'hard_linear_without_offset':
            machine_error = self.get_hard_linear_svm_w(subset_c)

        if self.svm_type == 'soft_linear_with_offset':
            machine_error = self.get_soft_linear_svm_w_b(subset_c)

        if self.svm_type == 'soft_linear_without_offset':
            machine_error = self.get_soft_linear_svm_w(subset_c)

        if self.svm_type == 'soft_kernel_with_offset':
            machine_error = self.get_soft_kernel_svm_w_b(subset_c)

        return - machine_error - c_S
