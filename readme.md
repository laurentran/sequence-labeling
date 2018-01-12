# Financial Default Prediction

This project uses Azure Machine Learning Workbench. Scikit-Learn, and Keras to perform sequence classification on financial transaction data, to predict whether someone is likely to default or not.  

This binary classification task is implemented in multiple ways:

1. Support Vector Machine
2. LSTM (Long Short-Term Memory)
3. Convolutional Neural Network

The script `preprocess.py` takes the input data and builds a 3D matrix to feed into the LSTM network of shape (samples, timesteps, features). 

On our dataset, the CNN implementation achieved the highest accuracy. 