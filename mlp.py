from keras.models import Sequential
from keras.layers import Dense
from Util import *

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

#train_x_orig, train_y, test_x_orig, test_y, classes

encoded_train_imgs, train_Y, encoded_test_imgs, test_Y = load_dataset()

input_shape = (feature_vector_length,)

# Create the model
model = Sequential()
model.add(Dense(350, input_shape=input_shape, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(num_classes, activation='sigmoid'))

# Configure the model and start training
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(encoded_train_imgs, train_Y, epochs=10, batch_size=250, verbose=1, validation_split=0.2)

# Test the model after training
test_results = model.evaluate(encoded_test_imgs, test_Y, verbose=1)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}%')