import pickle
from sklearn.svm import SVC
import numpy as np
import sys

file_name = sys.argv[1]
file = open('data/data_dict_' + file_name + '.pkl')

if file_name == 'Aptos':
	threshold = 1.8
if file_name == 'Messidor':
	threshold = 1.5
if file_name == 'Stare':
	threshold = 0.5

file = pickle.load(file)
YY = file['Y']
XX = file['X']

frac = (file['X'].shape[0]/10)*4
X = XX[frac:]
Y_discrete = YY[frac:]
X_te = XX[:frac]
Y_te_discrete = YY[:frac]
Y_tr = np.zeros(Y_discrete.shape)

for idx,label in enumerate(Y_discrete):
	if label>threshold:
		Y_tr[idx] = 1
	else:
		Y_tr[idx] = -1

Y_te = np.zeros(Y_te_discrete.shape)
for idx,label in enumerate(Y_te_discrete):
	if label>threshold:
		Y_te[idx] = 1
	else:
		Y_te[idx] = -1

val_frac = X.shape[0]/2
train = np.arange(val_frac)
val = np.arange(val_frac,2*val_frac)
def get_splits():
	yield train,val

from sklearn.model_selection import GridSearchCV

C_range = np.logspace(-5, 1, 17)

param_grid = dict(C=C_range)
grid = GridSearchCV(SVC(kernel='linear'), param_grid=param_grid, cv=get_splits())
grid.fit(X, Y_tr)
print("The best parameters are %s with a score of %0.2f"
      % (grid.best_params_, grid.best_score_))

train_res = grid.cv_results_['mean_train_score']
val_res = grid.cv_results_['mean_test_score']
landas = float(1)/(2.0*X.shape[0]*grid.cv_results_['param_C'])
C = grid.best_params_['C']
thelanda = float(1)/(2.0*train.shape[0]*C)
print thelanda

import matplotlib.pyplot as plt

test_res = []
my_val_res = []
plus_train_res = []
minus_train_res = []
plus_val_res = []
minus_val_res = []
plus_te_res = []
minus_te_res = []

train_X = X[train]
train_Y = Y_tr[train]
val_X = X[val]
val_Y = Y_tr[val]


train_plus = [idx for idx,_ in enumerate(train_Y) if train_Y[idx]==1]
train_minus = [idx for idx,_ in enumerate(train_Y) if train_Y[idx]==-1]
val_plus = [idx for idx,_ in enumerate(val_Y) if val_Y[idx]==1]
val_minus = [idx for idx,_ in enumerate(val_Y) if val_Y[idx]==-1]
test_plus = [idx for idx,_ in enumerate(Y_te) if Y_te[idx]==1]
test_minus = [idx for idx,_ in enumerate(Y_te) if Y_te[idx]==-1]

for C in grid.cv_results_['param_C']:
	model = SVC(kernel='linear',C=C)
	model.fit(train_X,train_Y)
	pred = model.predict(X_te)
	val_pred = model.predict(val_X)
	my_val_res.append(np.mean(val_pred==val_Y))
	test_res.append(np.mean(pred==Y_te))

landas=landas[::-1]
train_res=train_res[::-1]
test_res = test_res[::-1]
val_res = val_res[::-1]
my_val_res = my_val_res[::-1]

plt.plot(np.arange(landas.shape[0]),train_res)
plt.plot(np.arange(17),train_res,label='train_accuracy')
plt.plot(np.arange(17),val_res,label='val_accuracy')
plt.plot(np.arange(17),test_res,label='test_accuracy')

plt.legend()

for idx,landa in enumerate(landas):
	if landa>0:
		landas[idx] = str("{0:.2f}".format(landa))
	else:
		landas[idx] = str("{0:.4f}".format(landa))
plt.xticks(np.arange(17),landas)
plt.show()