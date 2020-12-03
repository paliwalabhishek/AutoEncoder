import numpy as np
import gzip
import Util

def preprocess_data(train_data, test_data):
    train_data = train_data.reshape(-1, 28, 28, 1)
    test_data = test_data.reshape(-1, 28, 28, 1)
    #scaling
    train_data = train_data / np.max(train_data)
    test_data = test_data / np.max(test_data)
    print(train_data.shape, test_data.shape)
    return train_data, test_data