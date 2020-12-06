import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from Util import plot_confusion_matrix
from sklearn import metrics
import time
import numpy as np


def load_dataset():
    # load dataset
    encoded_train_imgs = np.load('encoded_train_imgs_8.npy')
    train_Y = np.load('train_labels.npy')
    encoded_test_imgs = np.load('encoded_test_imgs_8.npy')
    test_Y = np.load('test_labels.npy')

    # encoded_train_imgs = np.expand_dims(encoded_train_imgs, axis=2)
    # encoded_test_imgs = np.expand_dims(encoded_test_imgs, axis=2)

    print("Training set (images) shape: {shape}".format(shape=encoded_train_imgs.shape))
    print("Test set (images) shape: {shape}".format(shape=encoded_test_imgs.shape))
    print("Training set (labels) shape: {shape}".format(shape=train_Y.shape))
    print("Test set (labels) shape: {shape}".format(shape=test_Y.shape))
    return encoded_train_imgs, train_Y, encoded_test_imgs, test_Y


batch_size = 250
feature_vector_length = 8
input_shape = (feature_vector_length,)
encoded_train_imgs, train_Y, encoded_test_imgs, test_Y = load_dataset()
# train_Y = np.argmax(train_Y, axis=1, out=None)
#encoded_train_imgs = encoded_train_imgs[:6000, :]
#train_Y = train_Y[:6000]
print("New Training set (images) shape: {shape}".format(shape=encoded_train_imgs.shape))
#test_Y = np.argmax(test_Y, axis=1, out=None)

train_dataset = tf.data.Dataset.from_tensor_slices((encoded_train_imgs, train_Y))
train_dataset = train_dataset.shuffle(buffer_size=1024).batch(batch_size)

def gausKer(y1, y2, s=0.01):
    return np.exp(-np.square(y1 - y2) / 2 * s ** 2) / (np.sqrt(2 * np.pi) * s ** 2)


def vm(d,x):
    sample_size = len(x)
    value = 0
    for k, v in d.items():
        sample_per_class = len(v)
        value = value + value * ((sample_per_class / sample_size) ** 2)
        for i in range(sample_size):
            for j in range(sample_size):
                value = value + gausKer(x[j], x[i])
    return value / (sample_size ** 2)


def vc(d,x):
    sample_size = len(x)
    value = 0
    for k, v in d.items():
        sample_per_class = len(v)
        value = value + value * (sample_per_class / sample_size)
        for i in range(sample_per_class):
            for j in range(sample_size):
                value = value + gausKer(x[j], v[i])
    return value / (sample_size ** 2)


def vj(d, x):
    sample_size = len(x)
    value = 0
    for k, v in d.items():
        sample_per_class = len(v)
        for i in range(sample_per_class):
            for j in range(sample_per_class):
                value = value + gausKer(v[j], v[i])
    return value / (sample_size ** 2)


def qmi_loss(train_data, y_batch_train):
    d = {}
    y = np.argmax(y_batch_train, axis=1, out=None)
    x = train_data.numpy()
    for i in range(len(y)):
        if y[i] not in d:
            d[y[i]] = [x[i]]
        else:
            d[y[i]].append(x[i])
    vj_value = vj(d,x)
    vc_value = vc(d,x)
    vm_value = vm(d,x)
    delta_qmi = vj_value - 2 * vc_value + vm_value
    res = delta_qmi / len(x_batch_train) ** 2
    return res


# Instantiate an optimizer.
optimizer = keras.optimizers.Adam()
# Instantiate a loss function.
loss_fn = keras.losses.CategoricalCrossentropy(from_logits=False)

inputs = keras.Input(shape=input_shape)
x1 = layers.Dense(64, activation="relu")(inputs)
x2 = layers.Dense(32, activation="relu")(x1)
outputs = layers.Dense(10, name="predictions")(x2)
model = keras.Model(inputs=inputs, outputs=outputs)
epochs = 20
start_time = time.time()
for epoch in range(epochs):
    print("\nStart of epoch %d" % (epoch,))
    for step, (x_batch_train, y_batch_train) in enumerate(train_dataset):
        logi = qmi_loss(x_batch_train, y_batch_train)
        with tf.GradientTape() as tape:
            logits = model(x_batch_train*logi)
            loss_value = loss_fn(y_batch_train, logits)
            if step % 250 == 0:
                print(
                    "Training loss (for one batch) at step %d: %.4f"
                    % (step, float(loss_value))
                )
        grads = tape.gradient(loss_value, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))


# Test the model after training
y_pred = model.predict(encoded_test_imgs)
pred = np.argmax(y_pred, axis=1, out=None)
true = np.argmax(test_Y, axis=1, out=None)
conf_matrix = metrics.confusion_matrix(y_true=true, y_pred=pred)
plot_confusion_matrix(cm=conf_matrix,
                      normalize=False,
                      title="MLP Confusion Matrix with 100 epochs "+str(feature_vector_length)+" bottleneck layer")