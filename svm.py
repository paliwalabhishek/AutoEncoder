import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, svm, metrics
from sklearn.metrics import confusion_matrix
import time
import pandas as pd
import seaborn as sns
import itertools

labeldict = {
    0: 'T-shirt/top',
    1: 'Trouser',
    2: 'Pullover',
    3: 'Dress',
    4: 'Coat',
    5: 'Sandal',
    6: 'Shirt',
    7: 'Sneaker',
    8: 'Bag',
    9: 'Ankle boot'
}

def plot_confusion_matrix(cm,
                          target_names,
                          title='Confusion matrix',
                          cmap=None,
                          normalize=True):
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(8, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.xlim(-0.5, len(np.unique(target_names)) - 0.5)
    plt.ylim(len(np.unique(target_names)) - 0.5, -0.5)
    plt.show()

## Data loading
encoded_train_imgs = np.load('encoded_train_imgs_50.npy')
train_Y = np.load('train_labels.npy')
encoded_test_imgs = np.load('encoded_test_imgs_50.npy')
test_Y = np.load('test_labels.npy')

print("Training set (images) shape: {shape}".format(shape=encoded_train_imgs.shape))
print("Test set (images) shape: {shape}".format(shape=encoded_test_imgs.shape))

train_Y = np.argmax(train_Y, axis=1, out=None)
test_Y = np.argmax(test_Y, axis=1, out=None)
print("Training set (labels) shape: {shape}".format(shape=train_Y.shape))
print("Test set (labels) shape: {shape}".format(shape=test_Y.shape))


non_linear_model = svm.SVC(C=10, gamma=0.01, kernel='rbf')

start_time = time.time()
non_linear_model.fit(encoded_train_imgs, train_Y)
print("--- %s seconds ---" % (time.time() - start_time))


y_pred = non_linear_model.predict(encoded_test_imgs)
print("accuracy:", metrics.accuracy_score(y_true=test_Y, y_pred=y_pred), "\n")

conf_matrix = metrics.confusion_matrix(y_true=test_Y, y_pred=y_pred)
plot_confusion_matrix(cm=conf_matrix,
                      normalize=False,
                      target_names=list(labeldict.values()),
                      title="Confusion Matrix")
#df_cm = pd.DataFrame(conf_matrix, index=list(labeldict.values()),  columns=list(labeldict.values()))
#sns.heatmap(df_cm, annot=True)
#plt.show()
