import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
# import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import scipy as sp
import sklearn
import random
import time
from sklearn import preprocessing, model_selection
import pickle
import joblib

from keras.models import Sequential
from keras.layers import Dense
from keras.utils import np_utils
from sklearn.preprocessing import LabelEncoder
from keras.utils.np_utils import to_categorical
from sklearn.utils import shuffle


# loading the dataset
data = pd.read_csv("final-draft1.csv")
data = shuffle(data)

i = 20
data_to_predict = data[:i].reset_index(drop = True)
# print(str(data_to_predict))
predict_chord = data_to_predict.chord
predict_chord = np.array(predict_chord)
prediction = np.array(data_to_predict.drop(['chord'],axis= 1))
# print(str(prediction))


data = data[i:].reset_index(drop = True)



X = data.drop(['chord'], axis = 1)
X = np.array(X)
Y = data['chord']

# Transform name species into numerical values
encoder = LabelEncoder()
encoder.fit(Y)
Y = encoder.transform(Y)
Y = np_utils.to_categorical(Y)

train_x, test_x, train_y, test_y = model_selection.train_test_split(X,Y,test_size = 0.1, random_state = 0)


input_dim = len(data.columns) - 1

model = Sequential()
model.add(Dense(10, input_dim = input_dim , activation = 'relu'))
model.add(Dense(20, activation = 'relu'))
model.add(Dense(25, activation = 'relu'))
model.add(Dense(30, activation = 'relu'))
model.add(Dense(35, activation = 'relu'))
# model.add(Dense(30, activation = 'relu'))
model.add(Dense(32, activation = 'softmax'))

model.compile(loss = 'categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'] )

model.fit(train_x, train_y, epochs = 350, batch_size = 15)

predictions = np.argmax(model.predict(prediction), axis=-1)
prediction_ = np.argmax(to_categorical(predictions), axis = 1)
prediction_ = encoder.inverse_transform(prediction_)

for i, j in zip(prediction_ , predict_chord):
    print( " the nn predict {}, and the chord to find is {}".format(i,j))
