import numpy.random as rand
from myutil import *

class Generate_human_error:

    def __init__(self, data_file):
        self.data = load_data(data_file)
        self.X = self.data['X']
        self.Y = self.data['Y']
        self.n, self.dim = self.data['X'].shape

    def estimated_uncertainty_generate_variable_human_prediction(self, Y, list_of_std, threshold, file_name):
        c = {}
        y_h = {}
        h = {}
        machine_h = {}
        Pr_H = {}
        doctors_label = {}
        doctor_label_thresholded = {}
        num_doctors = 4

        def get_num_category(y):
            return np.unique(y)

        def descrete_label(cont):
            if cont > threshold:
                return 1
            else:
                return -1

        cats = get_num_category(self.data['Y'])

        if file_name == 'Messidor':
            alpha = [[3, 3, 1, 1], [3, 2, 2, 1], [.5, .5, 5, 4], [0.1, 0.1, 4, 6]]
            hmap = {cats[0]: -1.0, cats[1]: -0.5, cats[2]: 0.5, cats[3]: 1}

        if file_name == 'Stare':
            alpha = [[3, 3, 2, 1, 1], [2, 7, .5, .5, .1], [.1, 1, 4, 3, 2], [1, 2, 3, 3, 1],
                     [.1, .1, 5, 5, 5]]
            hmap = {cats[0]: -1.0, cats[1]: -0.3, cats[2]: 0.3, cats[3]: 0.6, cats[4]: 1.0}

        if file_name == 'Aptos':
            alpha = [[4, 2, 1, 1, 1], [4, 4, 1, .5, .5], [.1, .1, 5, 4, 4], [.1, .1, 4, 5, 4],
                     [.1, .1, 4, 4, 5]]
            hmap = {cats[0]: -1.0, cats[1]: -0.5, cats[2]: 0.3, cats[3]: 0.6, cats[4]: 1.0}

        for std in list_of_std:
            machine_h[str(std)] = np.zeros(shape=Y.shape)
            y_h[str(std)] = np.zeros(shape=Y.shape, dtype='int')
            c[str(std)] = np.zeros(shape=Y.shape)
            h[str(std)] = np.zeros(shape=Y.shape)
            Pr_H[str(std)] = np.zeros(shape=Y.shape)
            doctors_label[str(std)] = np.zeros((Y.shape[0], num_doctors))
            doctor_label_thresholded[str(std)] = np.zeros((Y.shape[0], num_doctors))

            for idx, label in enumerate(Y):
                thresholded_label = descrete_label(label)
                trueindex = np.argwhere(label == cats)[0][0]
                prob_vector = np.random.dirichlet(alpha[trueindex], num_doctors)
                h_sample = []

                for prob_idx, doctor_prob in enumerate(prob_vector):
                    doctors_label[str(std)][idx][prob_idx] = (np.random.choice(cats, 1, p=doctor_prob)[0])
                    h_sample.append(hmap[doctors_label[str(std)][idx][prob_idx]])

                for prob_idx, probability in enumerate(prob_vector[0]):
                    c[str(std)][idx] += probability * np.maximum(0, 1 - (
                            hmap[cats[prob_idx]] * thresholded_label))

                h[str(std)][idx] = np.mean(h_sample)

                doctor_label_thresholded[str(std)][idx] = [descrete_label(doc_label) for doc_label in
                                                           doctors_label[str(std)][idx]]

                A_idx = []
                while len(A_idx) < num_doctors / 2:
                    num = np.random.randint(0, num_doctors)
                    if num not in A_idx:
                        A_idx.append(num)

                A = np.array(
                    [A_label for doc_idx, A_label in enumerate(doctors_label[str(std)][idx]) if doc_idx in A_idx])
                B = np.array(
                    [B_label for doc_idx, B_label in enumerate(doctors_label[str(std)][idx]) if doc_idx not in A_idx])

                A_mean = descrete_label(np.mean(A))
                B_mean = descrete_label(np.mean(B))

                if A_mean != B_mean:
                    Pr_H[str(std)][idx] = 1  # disagreement
                else:
                    Pr_H[str(std)][idx] = 0  # agreement

                final_label_thresholded = doctor_label_thresholded[str(std)][idx][0]
                y_h[str(std)][idx] = final_label_thresholded

        return c, h, y_h, Pr_H

    def get_human_error(self, list_of_std, threshold, file_name):

        self.data['c'], self.data['h'], self.data['y_h'], self.data['Pr_H'] = \
            self.estimated_uncertainty_generate_variable_human_prediction(self.data['Y'], list_of_std, threshold,
                                                                          file_name)

    def save_data(self, data_file):
        save(self.data, data_file)


def generate_human_error(file_name_list):
    datasets = ['Messidor' 'Stare', 'Aptos']

    for file_name in file_name_list:
        assert file_name in datasets

        list_of_std = [1]

        data_file = 'data/data_dict_' + file_name
        obj = Generate_human_error(data_file)

        if file_name == 'Stare':
            threshold = 0.5
        if file_name == 'Aptos':
            threshold = 1.8
        if file_name == 'Messidor':
            threshold = 1.5

        obj.get_human_error(list_of_std, threshold, file_name)

        if os.path.exists('data/data_dict_' + file_name):
            os.path.remove('data/data_dict_' + file_name)

        obj.save_data('data/data_dict_' + file_name)


def main():
    file_name_list = sys.argv[1:]
    generate_human_error(file_name_list)


if __name__ == "__main__":
    main()
