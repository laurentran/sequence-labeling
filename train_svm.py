import pickle
import sys
import os

import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve, roc_curve, auc

from azureml.logging import get_azureml_logger
from azureml.dataprep.package import run

run_logger = get_azureml_logger()

os.makedirs('./outputs', exist_ok=True)

# read dataset as dataframe
print('-------Training model-------')
data = run('Weekly.dprep', dataflow_idx=0, spark=False)
print('Dataset shape: {}'.format(data.shape))

# read features and labels
X = data.iloc[:,2:3574]
Y = data['Goal']

# normalize data
X = preprocessing.normalize(X, norm='l2')

train_X, test_X, train_Y, test_Y = train_test_split(X, Y, train_size = 0.7, random_state=1, stratify=Y)

# train SVM 
clf = svm.SVC(probability=True)
clf.fit(train_X, train_Y)

# score training set
train_acc = clf.score(train_X, train_Y)
print ('training accuracy: {}'.format(train_acc))

# score test set
test_acc = clf.score(test_X, test_Y)
print ('validation accuracy: {}'.format(test_acc))

run_logger.log("Training accuracy", train_acc)
run_logger.log("Validation accuracy", test_acc)


# evaluate metrics
predict_Y = clf.predict_proba(test_X)
fpr, tpr, thresholds2 = roc_curve(test_Y, predict_Y[:,1], pos_label=0)

roc_auc = auc(fpr, tpr)

#plot roc curve
fig = plt.figure(figsize=(6, 5), dpi=75)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'r--')
plt.xlim([-0.1,1.1])
plt.ylim([-0.1,1.1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
fig.savefig("./outputs/roc.png", bbox_inches='tight')


