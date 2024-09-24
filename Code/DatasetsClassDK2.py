import pickle
import os
import numpy as np
from tensorflow.keras.utils import Sequence
from scipy.signal.windows import tukey


class DataGeneratorPickles(Sequence):

    def __init__(self, data_dir, filename, input_size, conditioning_size, batch_size=10):
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
        self.input_size = input_size
        self.conditioning_size = conditioning_size

        # prepare the input, taget and conditioning matrix
        self.x, self.y, self.z, rep, lim = self.prepareXYZ(data_dir, filename)

        self.training_steps = (lim // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):

        # load all the audio files
        file_data = open(os.path.normpath(
            '/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:1, :], dtype=np.float32)
        y = np.array(Z['y'][:1, :], dtype=np.float32)

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
        x = x.reshape(-1)
        y = y.reshape(-1)

        # how many iteration it is needed
        N = int((x.shape[0] - self.input_size) / self.batch_size)-1
        # how many total samples is the audio
        lim = int(N * self.batch_size) + self.input_size - 1
        x = x[:lim]
        y = y[:lim]

        # loading the conditioning values
        if self.conditioning_size != 0:
            z = np.array(Z['z'], dtype=np.float32)
            z = np.repeat(z, rep, axis=0)
        else:
            z = None

        return x, y, z, rep, lim

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.input_size-1, self.x.shape[0]+1)

    def __len__(self):
        # compute the itneeded number of iteration before conclude one epoch
        return int(self.x.shape[0] / self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # Initializing input, target, and conditioning batches
        X = np.empty((self.batch_size, self.input_size))
        Y = np.empty((self.batch_size, 1))

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        c = 0

        if self.conditioning_size != 0:
            Z = np.empty((self.batch_size, self.conditioning_size))
            # fill the batches
            for t in range(indices[0], indices[-1]+1, 1):
                X[c, :] = np.array(self.x[t - self.input_size+1: t+1])
                Y[c, :] = np.array(self.y[t])
                Z[c, :] = np.array(self.z[t])
                c = c + 1

            return [Z, X], Y

        else:
            # fill the batches
            for t in range(indices[0], indices[-1] + 1, 1):
                X[c, :] = np.array(self.x[t - self.input_size+1: t+1])
                Y[c, :] = np.array(self.y[t])

                c = c + 1

            return X, Y


class DataGeneratorPicklesStudent(Sequence):

    def __init__(self, data_dir, filename, input_size, conditioning_size, batch_size=10):
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
        self.input_size = input_size
        self.conditioning_size = conditioning_size

        # prepare the input, taget and conditioning matrix
        self.x, self.yh, self.z, rep, lim, self.weights = self.prepareXYZ(
            data_dir, filename)

        self.training_steps = (lim // self.batch_size)
        self.on_epoch_end()

    def prepareXYZ(self, data_dir, filename):

        # load all the audio files
        file_data = open(os.path.normpath(
            '/'.join([data_dir, filename])), 'rb')
        Z = pickle.load(file_data)
        x = np.array(Z['x'][:1, :], dtype=np.float32)
        #y = np.array(Z['y'][:1, :], dtype=np.float32)
        yh = np.array(Z['yh'], dtype=np.float32)
        weights = Z['w'] # those are the weight of the output layer

        # if input is shared to all the targets, it is repeat accordingly to the number of target audio files
        #if x.shape[0] == 1:
        #    x = np.repeat(x, yh.shape[0], axis=0)

        # windowing the signals in order to avoid misalignments
        x = x * np.array(tukey(x.shape[1], alpha=0.000005),
                         dtype=np.float32).reshape(1, -1)
        #y = y * np.array(tukey(x.shape[1], alpha=0.000005),
        #                 dtype=np.float32).reshape(1, -1)

        # reshape to one dimension
        rep = x.shape[1]
        x = x.reshape(-1)
        #y = y.reshape(-1)

        # how many iteration it is needed
        N = int((x.shape[0] - self.input_size) / self.batch_size)-1
        # how many total samples is the audio
        lim = int(N * self.batch_size) + self.input_size - 1
        x = x[:lim]
        #y = y[:lim]

        # loading the conditioning values
        if self.conditioning_size != 0:
            z = np.array(Z['z'], dtype=np.float32)
            z = np.repeat(z, rep, axis=0)
        else:
            z = None

        return x, yh, z, rep, lim, weights

    def on_epoch_end(self):
        # create/reset the vector containing the indices of the batches
        self.indices = np.arange(self.input_size-1, self.x.shape[0]+1)

    def __len__(self):
        # compute the itneeded number of iteration before conclude one epoch
        return int(self.x.shape[0] / self.batch_size)

    def __call__(self):
        for i in range(self.__len__()):
            yield self.__getitem__(i)
            if i == self.__len__() - 1:
                self.on_epoch_end()

    def __getitem__(self, idx):
        # Initializing input, target, and conditioning batches
        X = np.empty((self.batch_size, self.input_size))
        YH = np.empty((self.batch_size, 8))

        # get the indices of the requested batch
        indices = self.indices[idx*self.batch_size:(idx+1)*self.batch_size]
        c = 0

        if self.conditioning_size != 0:
            Z = np.empty((self.batch_size, self.conditioning_size))
            # fill the batches
            for t in range(indices[0], indices[-1]+1, 1):
                X[c, :] = np.array(self.x[t - self.input_size+1: t+1])
                Z[c, :] = np.array(self.z[t])
                YH[c, :] = np.array(self.yh[t])
                c = c + 1

            return [Z, X], YH

        else:
            # fill the batches
            for t in range(indices[0], indices[-1] + 1, 1):
                X[c, :] = np.array(self.x[t - self.input_size+1: t+1])
                YH[c, :] = np.array(self.yh[t])
                c = c + 1

            return X, YH