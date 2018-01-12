# Sequence Labeling with Traditional and DL Models for Financial Default Prediction

This project uses Azure Machine Learning Workbench. Scikit-Learn, and Keras to perform sequence classification on financial transaction data, to predict whether someone is likely to default. 

This binary classification task is implemented in multiple ways:

1. [Support Vector Machine](https://github.com/laurentran/financial-default-prediction/blob/master/train_svm.py): Traditional ML approach with feature engineering, using SVM to handle high dimensionality and sparse feature vectors.
2. [Long Short-Term Memory](https://github.com/laurentran/financial-default-prediction/blob/master/train_lstm.py): Deep learning approach using LSTM, a natural choice for sequence labeling with it's ability to learn long-term dependencies.
3. [Convolutional Neural Network](https://github.com/laurentran/financial-default-prediction/blob/master/train_cnn.py): Extending CNN beyond it's traditional image-based tasks, using Convolution 1D to find spatial patterns in the sequential data.  With our dataset, CNN achieved highest accuracy and lower training times than LSTM.

The script [preprocess.py](https://github.com/laurentran/financial-default-prediction/blob/master/preprocess.py) takes the input data and builds a 3D matrix to feed into the LSTM network of shape (samples, timesteps, features).
