from keras.models import Sequential
from keras.layers import Dense
from Util import *
import keras.backend as K
import math
import tensorflow as tf

feature_vector_length = 5
num_classes = 10


def load_dataset():
    # load dataset
    encoded_train_imgs = np.load('encoded_train_imgs_5.npy')
    train_Y = np.load('train_labels_5.npy')
    encoded_test_imgs = np.load('encoded_test_imgs_5.npy')
    test_Y = np.load('test_labels_5.npy')

    # encoded_train_imgs = np.expand_dims(encoded_train_imgs, axis=2)
    # encoded_test_imgs = np.expand_dims(encoded_test_imgs, axis=2)

    print("Training set (images) shape: {shape}".format(shape=encoded_train_imgs.shape))
    print("Test set (images) shape: {shape}".format(shape=encoded_test_imgs.shape))
    print("Training set (labels) shape: {shape}".format(shape=train_Y.shape))
    print("Test set (labels) shape: {shape}".format(shape=test_Y.shape))
    return encoded_train_imgs, train_Y, encoded_test_imgs, test_Y


# train_x_orig, train_y, test_x_orig, test_y, classes

encoded_train_imgs, train_Y, encoded_test_imgs, test_Y = load_dataset()

input_shape = (feature_vector_length,)


def gaussianKernel(x1, x2, sigma=0.01 * math.sqrt(2)):
    sim = K.exp(- K.abs((x1 - x2) ** 2) / float(2(sigma ** 2)))
    return sim

def mee(y_true, y_pred):

    e = y_true - y_pred
    e2 = K.repeat(e, K.shape(e)[0])
    m = tf.cast(tf.shape(y_true)[0], 'float32')
    s = 0.01
    pi = tf.cast(np.pi, 'float32')
    return - K.sum(K.exp(-K.square(e - e2) / 2*s**2) / (K.sqrt(2*pi)*s*m**2))


def qmi(y_true, y_pred):
    [x1_data, x2_data] = [y_true[:, 0], y_true[:, 1]]
    value = gaussianKernel(y_pred, y_true)
    print('current loss', value)
    loss = K.sum(value)
    print('sum loss', loss)
    return -loss


# Create the model
model = Sequential()
model.add(Dense(350, input_shape=input_shape, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(feature_vector_length, activation='sigmoid'))

# Configure the model and start training
model.compile(loss=mee, optimizer='adam', metrics=['accuracy'])
history = model.fit(encoded_train_imgs, train_Y, epochs=10, batch_size=250, verbose=1, validation_split=0.2)

# Test the model after training
test_results = model.evaluate(encoded_test_imgs, test_Y, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')

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
