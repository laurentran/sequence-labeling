import numpy as np
import pandas as pd

from keras.models import Sequential
from keras.layers import Dense, Conv1D, Flatten, MaxPool1D, Dropout, Activation

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run
run_logger = get_azureml_logger()

epochs = 100
batch_size = 10
data_path = 'data/Weekly.csv'

# # read dataset as dataframe
data = pd.read_csv(data_path)

# read features and labels
X = data.iloc[:,2:3574]
Y = data['Goal']

# convert to numpy array
X = X.values

# normalize data
X = preprocessing.normalize(X, norm='l2')

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size = 0.7, random_state=1, stratify=Y)

# reshape into correct dimensions to input into cnn
train_X = train_X.reshape(140,3572,1)
test_X = test_X.reshape(60,3572,1)

# build network layers
model = Sequential()
model.add(Conv1D(128,3,input_shape=(3572, 1)))
model.add(Activation('relu'))
model.add(MaxPool1D(pool_size=2))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())
model.fit(train_X, train_Y, validation_data=(test_X, test_Y), epochs=epochs, batch_size=batch_size)

# score model and log accuracy and parameters
scores = model.evaluate(test_X, test_Y, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

run_logger.log("Epochs", epochs)
run_logger.log("Batch Size", batch_size)
run_logger.log("Accuracy", scores[1]*100)
