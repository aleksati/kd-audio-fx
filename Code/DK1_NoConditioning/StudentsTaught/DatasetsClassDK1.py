import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey
from Utils import filterAudio


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, mini_batch_size, input_size, batch_size=10):
        """
        Initializes a data generator object
          :param data_dir: the directory in which data are stored
          :param filename: the name of the dataset
          :param input_size: the inpput size
          :param cond: the number of conditioning values
          :param batch_size: The size of each batch returned by __getitem__
        """
        self.data_dir = data_dir
        self.filename = filename
        self.batch_size = batch_size
        self.mini_batch_size = mini_batch_size
        self.input_size = input_size

        # prepare the input, taget and conditioning matrix
        self.x, self.y, self.z, rep, lim = self.prepareXYZ(data_dir, filename)

        self.max_1 = (self.x.shape[1] // self.mini_batch_size) - 1
        self.max = (self.max_1 // self.batch_size) - 1

        self.training_steps = self.max

        self.training_steps = (lim // self.batch_size)

        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):

        # load all the audio files
        file_data = open(os.path.normpath(
            '/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:1, :], dtype=np.float32)
        y = np.array(Z['y'][:1, :], dtype=np.float32)
        y = filterAudio(y)

        # if input is shared to all the targets, it is repeat accordingly to the number of target audio files
        if x.shape[0] == 1:
            x = np.repeat(x, y.shape[0], axis=0)

        # windowing the signals in order to avoid misalignments
        x = x * np.array(tukey(x.shape[1], alpha=0.000005),
                         dtype=np.float32).reshape(1, -1)
        y = y * np.array(tukey(x.shape[1], alpha=0.000005),
                         dtype=np.float32).reshape(1, -1)

        # reshape to one dimension
        rep = x.shape[1]
        x = x.reshape(1, -1)
        y = y.reshape(1, -1)

        # how many iteration it is needed
        N = int((x.shape[0] - self.input_size) / self.batch_size) - 1
        # how many total samples is the audio
        lim = int(N * self.batch_size) + self.input_size - 1
        x = x[:, :lim]
        y = y[:, :lim]

        # loading the conditioning values

        z = None

        return x, y, z, rep, lim

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(0, self.x.shape[1])

    def __len__(self):
        # compute the needed number of iteration before conclude one epoch
        return int(self.max)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):

        # get the indices of the requested batch
        indices = self.indices[
                  idx * self.mini_batch_size * self.batch_size:(idx + 1) * self.mini_batch_size * self.batch_size]

        # fill the batches
        X = np.array(self.x[0, indices]).reshape(self.batch_size, self.mini_batch_size, 1)
        Y = np.array(self.y[0, indices]).reshape(self.batch_size, self.mini_batch_size, 1)

        return X, Y