import numpy as np

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing import sequence

from azureml.logging import get_azureml_logger
run_logger = get_azureml_logger()

epochs = 100
batch_size = 10
time_steps = 153
features = 37

features_path = 'data/sequence_data.npy'
labels_path = 'data/sequence_labels.npy'

# load data from .npy files into numpy arrays
X = np.load(features_path)
Y = np.load(labels_path)

# split data into training and test sets 
# this dataset is a 3D matrix, so sklearn train_test_split will not work
# split your data as it makes sense here
train_X1 = X[0:69,:,:]
train_X2 = X[130:199,:,:]
train_X = np.concatenate((train_X1,train_X2), axis=0)
test_X = X[70:129,:,:]
train_Y1 = Y[0:69,:]
train_Y2 = Y[130:199,:]
train_Y = np.concatenate((train_Y1,train_Y2), axis = 0)
test_Y = Y[70:129,:]

# build LSTM layers
model = Sequential()
model.add(LSTM(100, dropout=0.2, input_shape=(time_steps, features)))
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
