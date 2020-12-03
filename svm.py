import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix

## Data loading
encoded_train_imgs = np.load('encoded_train_imgs_5.npy')
train_Y = np.load('train_labels_5.npy')
encoded_test_imgs = np.load('encoded_test_imgs_5.npy')
test_Y = np.load('test_labels_5.npy')

print("Training set (images) shape: {shape}".format(shape=encoded_train_imgs.shape))
print("Test set (images) shape: {shape}".format(shape=encoded_test_imgs.shape))

train_Y = np.argmax(train_Y, axis=1, out=None)
test_Y = np.argmax(test_Y, axis=1, out=None)
print("Training set (labels) shape: {shape}".format(shape=train_Y.shape))
print("Test set (labels) shape: {shape}".format(shape=test_Y.shape))
"""
## Linear model
model_linear = svm.SVC(kernel='linear')
model_linear.fit(encoded_train_imgs, train_Y)
y_pred = model_linear.predict(encoded_test_imgs)
# accuracy
print("accuracy:", metrics.accuracy_score(y_true=test_Y, y_pred=y_pred), "\n")
# cm
print(metrics.confusion_matrix(y_true=test_Y, y_pred=y_pred))
"""
# non-linear model
# model
## Hyperparameter Tuning
"""
https://www.kaggle.com/nishan192/mnist-digit-recognition-using-svm
"""

print("RUNNING for SVM non-linear model for batch size = 5, C=10, gamma = 0.01")
non_linear_model = svm.SVC(C=10, gamma=0.01, kernel='rbf')
# fit
non_linear_model.fit(encoded_train_imgs, train_Y)
# predict
y_pred = non_linear_model.predict(encoded_test_imgs)
# accuracy
print("accuracy:", metrics.accuracy_score(y_true=test_Y, y_pred=y_pred), "\n")
# cm
print(metrics.confusion_matrix(y_true=test_Y, y_pred=y_pred))
