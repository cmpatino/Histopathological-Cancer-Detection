import cv2
import numpy as np
from keras.utils import Sequence

DATA_PATH = '../input/'
TRAIN_DIR = DATA_PATH + 'train/'
TEST_DIR = DATA_PATH + 'test/'

CROP_SIZE = 64
ORIGINAL_SIZE = 96


class DataGenerator(Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, labels, path, batch_size=32, dim=(32, 32),
                 n_channels=3, n_classes=10, shuffle=True):
        'Initialization'
        self.dim = dim
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.path = path
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y = self.__data_generation(list_IDs_temp)

        return X, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle is True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels))
        y = np.empty((self.batch_size), dtype=int)

        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            # Store sample
            bgr_img = cv2.imread(self.path + ID + '.tif')
            b, g, r = cv2.split(bgr_img)
            rgb_img = cv2.merge([r, g, b])
            X[i, ] = rgb_img/255

            # Store class
            y[i] = self.labels[ID]

        return X, y
