from keras.models import Sequential
from keras.layers import Dense
from Util import plot_confusion_matrix
from sklearn import metrics
import time
import numpy as np
import matplotlib.pyplot as plt

feature_vector_length = 8
num_classes = 10
def load_dataset():
    # load dataset
    encoded_train_imgs = np.load('encoded_train_imgs_'+str(feature_vector_length)+'.npy')
    train_Y = np.load('train_labels.npy')
    encoded_test_imgs = np.load('encoded_test_imgs_'+str(feature_vector_length)+'.npy')
    test_Y = np.load('test_labels.npy')

    # encoded_train_imgs = np.expand_dims(encoded_train_imgs, axis=2)
    # encoded_test_imgs = np.expand_dims(encoded_test_imgs, axis=2)

    print("Training set (images) shape: {shape}".format(shape=encoded_train_imgs.shape))
    print("Test set (images) shape: {shape}".format(shape=encoded_test_imgs.shape))
    print("Training set (labels) shape: {shape}".format(shape=train_Y.shape))
    print("Test set (labels) shape: {shape}".format(shape=test_Y.shape))
    return encoded_train_imgs, train_Y, encoded_test_imgs, test_Y

encoded_train_imgs, train_Y, encoded_test_imgs, test_Y = load_dataset()

input_shape = (feature_vector_length,)

# Create the model
model = Sequential()
model.add(Dense(350, input_shape=input_shape, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
start_time = time.time()
history = model.fit(encoded_train_imgs, train_Y, epochs=50, batch_size=50, verbose=2, validation_split=0.2)
print("--- %s seconds ---" % ((time.time() - start_time)+14400))

# Test the model after training
y_pred = model.predict(encoded_test_imgs)
pred = np.argmax(y_pred, axis=1, out=None)
true = np.argmax(test_Y, axis=1, out=None)
conf_matrix = metrics.confusion_matrix(y_true=true, y_pred=pred)
plot_confusion_matrix(cm=conf_matrix,
                      normalize=False,
                      title="QMI loss MLP Confusion Matrix with 100 epochs "+str(feature_vector_length)+" bottleneck layer")

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
