import numpy as np
from tensorflow.keras.datasets import fashion_mnist
from sklearn.model_selection import train_test_split
import keras
from matplotlib import pyplot as plt
from keras.models import Model
from keras.optimizers import RMSprop
from keras.layers import Input,Dense,Flatten,Dropout,merge,Reshape,Conv2D,MaxPooling2D,UpSampling2D,Conv2DTranspose
from keras.layers.normalization import BatchNormalization
from keras.models import Model,Sequential
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adadelta, RMSprop,SGD,Adam
from keras import regularizers
from keras import backend as K
from keras.utils import to_categorical

code_size = 5
epochs = 100
batch_size = 200
nb_classes = 10

def preprocess_data(train_data, test_data):
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    #scaling
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)
    print(train_data.shape, test_data.shape)
    return train_data, test_data

((trainX, trainY), (testX, testY)) = fashion_mnist.load_data()
trainX = trainX.reshape(trainX.shape[0], 1, 28, 28)
testX = testX.reshape(testX.shape[0], 1, 28, 28)
trainX = trainX.astype("float32").reshape((-1, 784))
testX = testX.astype("float32").reshape((-1, 784))
trainX /= 255
testX /= 255
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
print("Training set (images) shape: {shape}".format(shape=trainX.shape))
print("Test set (images) shape: {shape}".format(shape=testX.shape))

Y_train = np_utils.to_categorical(trainY, nb_classes)
Y_test = np_utils.to_categorical(testY, nb_classes)

#train_X,valid_X,train_Y,valid_Y = train_test_split(trainX, Y_train, test_size=0.2, random_state=13)

for code_size in range(50, 60, 10):
    enc = Sequential()
    enc.add(Dense(500, activation='tanh', input_dim=784))
    enc.add(BatchNormalization())
    enc.add(Dense(200, activation='tanh'))
    enc.add(BatchNormalization())
    enc.add(Dense(code_size, activation='linear'))

    dec = Sequential()
    dec.add(Dense(200, input_dim=code_size, activation='tanh'))
    dec.add(Dense(500, activation='tanh'))
    dec.add(Dense(784, activation='sigmoid'))

    model = Sequential()
    model.add(enc)
    model.add(dec)

    model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

    history = model.fit(trainX, trainX.reshape((-1, 784)), batch_size=batch_size, verbose=2, epochs=epochs, validation_split=0.2)

    encoded_train_imgs = np.array(enc.predict(trainX))
    encoded_test_imgs = np.array(enc.predict(testX))

    ## Saving Data
    np.save("encoded_train_imgs_" + str(code_size) + ".npy", encoded_train_imgs)
    np.save("encoded_test_imgs_" + str(code_size) + ".npy", encoded_test_imgs)

"""
decoded_imgs = dec.predict(encoded_imgs)
n = 10  # How many digits we will display
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(2, n, i + 1)
    plt.imshow(valid_X[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    # Display reconstruction
    ax = plt.subplot(2, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(28, 28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
plt.show()
"""
"""
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
"""
