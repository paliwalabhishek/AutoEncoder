import numpy as np
from numpy import mean
from numpy import std
from matplotlib import pyplot
from sklearn.model_selection import KFold
from keras.models import Sequential
from keras.layers import Conv1D, Dropout
from keras.layers import MaxPooling1D
from keras.layers import Dense
from keras.layers import Flatten
from keras.optimizers import SGD


# load train and test dataset
def load_dataset():
    # load dataset
    encoded_train_imgs = np.load('encoded_train_imgs_5.npy')
    train_Y = np.load('train_labels_5.npy')
    encoded_test_imgs = np.load('encoded_test_imgs_5.npy')
    test_Y = np.load('test_labels_5.npy')

    encoded_train_imgs = np.expand_dims(encoded_train_imgs, axis=2)
    encoded_test_imgs = np.expand_dims(encoded_test_imgs, axis=2)

    print("Training set (images) shape: {shape}".format(shape=encoded_train_imgs.shape))
    print("Test set (images) shape: {shape}".format(shape=encoded_test_imgs.shape))

#    train_Y = np.argmax(train_Y, axis=1, out=None)
#    test_Y = np.argmax(test_Y, axis=1, out=None)
    print("Training set (labels) shape: {shape}".format(shape=train_Y.shape))
    print("Test set (labels) shape: {shape}".format(shape=test_Y.shape))
    return encoded_train_imgs, train_Y, encoded_test_imgs, test_Y


# define cnn model
def define_model(code_size):
    n_timesteps, n_features, n_outputs = 1, code_size, 10
    model = Sequential()
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu', input_shape=(n_features, n_timesteps)))
    model.add(Conv1D(filters=64, kernel_size=1, activation='relu'))
    model.add(Dropout(0.5))
    model.add(MaxPooling1D(pool_size=1))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(n_outputs, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform', input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Flatten())
    model.add(Dense(100, activation='relu', kernel_initializer='he_uniform'))
    model.add(Dense(10, activation='softmax'))
    # compile model
    opt = SGD(lr=0.01, momentum=0.9)
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    """
    return model


# evaluate a model using k-fold cross-validation
def evaluate_model(dataX, dataY, testX, testY, code_size, n_folds=5,):
    epochs = 15
    batch_size = 100
    model = define_model(code_size)
    history = model.fit(dataX, dataY, epochs=epochs, batch_size=batch_size)
    _, acc = model.evaluate(testX, testY, verbose=0)

    scores, histories = list(), list()
    scores.append(acc)
    histories.append(history)
    """
    # prepare cross validation
    kfold = KFold(n_folds, shuffle=True, random_state=1)
    # enumerate splits
    for train_ix, test_ix in kfold.split(dataX):
        # define model
        model = define_model(code_size)
        # select rows for train and test
        trainX, trainY, testX, testY = dataX[train_ix], dataY[train_ix], dataX[test_ix], dataY[test_ix]
        # fit model
        history = model.fit(trainX, trainY, epochs=10, batch_size=32, validation_data=(testX, testY), verbose=0)
        # evaluate model
        _, acc = model.evaluate(testX, testY, verbose=0)
        print('> %.3f' % (acc * 100.0))
        # stores scores
        scores.append(acc)
        histories.append(history)
    """
    return scores, histories


# plot diagnostic learning curves
def summarize_diagnostics(histories):
    for i in range(len(histories)):
        # plot loss
        pyplot.subplot(2, 1, 1)
        pyplot.title('Cross Entropy Loss')
        pyplot.plot(histories[i].history['loss'], color='blue', label='train')
        #pyplot.plot(histories[i].history['val_loss'], color='orange', label='test')
        # plot accuracy
        pyplot.subplot(2, 1, 2)
        pyplot.title('Classification Accuracy')
        pyplot.plot(histories[i].history['accuracy'], color='blue', label='train')
        #pyplot.plot(histories[i].history['val_accuracy'], color='orange', label='test')
    pyplot.show()


# summarize model performance
def summarize_performance(scores):
    # print summary
    print('Accuracy: mean=%.3f std=%.3f, n=%d' % (mean(scores) * 100, std(scores) * 100, len(scores)))
    # box and whisker plots of results
    pyplot.boxplot(scores)
    pyplot.show()


# run the test harness for evaluating a model
def run_test_harness():
    code_size = 5
    # load dataset
    encoded_train_imgs, train_Y, encoded_test_imgs, test_Y = load_dataset()
    # evaluate model
    scores, histories = evaluate_model(encoded_train_imgs, train_Y, encoded_test_imgs, test_Y, code_size)
    # learning curves
    summarize_diagnostics(histories)
    # summarize estimated performance
    summarize_performance(scores)


# entry point, run the test harness
run_test_harness()