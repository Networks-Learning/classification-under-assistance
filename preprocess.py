import numpy as np 
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt
import random
import os
import pickle
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder

# from keras.optimizers import Adam
# from keras.models import Sequential
# from keras.layers.core import Activation
# from keras.layers.core import Flatten
# from keras.layers.core import Dense
# 
# from keras.utils.np_utils import to_categorical
# from keras import backend as K
# import tensorflow as tf
################ First Step
# from keras.preprocessing.image import img_to_array, load_img
# base_dir = 'images/'
# dataset = {'X':{},'Y':{},'second_Y':{},'idx':{}}
# data = []
# labels = []
# indexes = []
# second_labels = []
# cnt = 0

# for dirs in os.listdir(base_dir):
#     print dirs
#     annotation_path =  base_dir + dirs + os.sep + 'Annotation_'+dirs+'.xls'
#     annotations = pd.read_excel(annotation_path)
#     image_names = annotations['Image name']
#     for label,second_label,image_name in zip(annotations['Retinopathy grade'],annotations['Risk of macular edema '],image_names):
#         image = load_img(base_dir + dirs+os.sep+image_name, target_size=(224,224))
#         image = img_to_array(image)
#         data.append(image)
#         labels.append(label)
#         second_labels.append(second_label)
#         indexes.append(image_name)
#         print cnt
#         cnt+=1

# labels = np.array(labels)
# indexes = np.array(indexes)
# second_labels = np.array(second_labels)
# data = np.array(data)
# data /= 255.0

# dataset['X'] = data
# dataset['Y'] = labels
# dataset['idx'] = indexes
# dataset['second_Y'] = second_labels
# print data.shape
# print labels.shape
# print indexes.shape
# with open('messidor_new_raw.pkl', 'wb') as f:
#     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

########################## Second step
# from keras.applications.vgg16 import VGG16
# file = open('messidor_new_raw.pkl')
# file = pickle.load(file)
# print file.keys()
# data = file['X']
# print data.shape
# from keras.applications.vgg16 import preprocess_input

# trained_model = VGG16(input_shape=(224,224,3))
# print trained_model.summary()
# trained_model.layers.pop()
# from keras.models import Model
# model = Model(inputs=trained_model.inputs, outputs=trained_model.layers[-1].output)

# features = np.zeros((data.shape[0],4096))
# feature_labels = np.zeros((data.shape[0]))
# for idx,img in enumerate(data):
# 	img = img.reshape(1,224,224,3)
# 	img = preprocess_input(img)
# 	print idx
# 	features[idx] = model.predict(img)


# dataset = {'X':features,'Y':file['Y'],'second_Y':file['second_Y'],'idx':file['idx']}

# with open('messidor_features.pkl', 'wb') as f:
#     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)

#################################### Third step
# file = open('messidor_features.pkl')
# file = pickle.load(file)
# X = file['X']
# Y = file['Y']
# second_Y = file['second_Y']
# indexes = file['idx']

# rnd = np.random.permutation(X.shape[0])
# X, Y, second_Y, indexes = X[rnd],Y[rnd],second_Y[rnd],indexes[rnd]

# num_features=50
# from sklearn.preprocessing import StandardScaler
# from sklearn.decomposition import PCA

# pca = PCA(n_components=num_features)
# sc = StandardScaler()


# X = sc.fit_transform(X)
# X = pca.fit_transform(X)

# dataset = {'X':X, 'Y':Y, 'second_Y':second_Y, 'idx':indexes}
# with open('data_dict_new_messidor' + '.pkl', 'wb') as f:
#     pickle.dump(dataset, f, pickle.HIGHEST_PROTOCOL)
