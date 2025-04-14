from Metrics import ESR, RMSE, STFT_loss
import numpy as np
import tensorflow as tf
import os
import glob
from scipy.io import wavfile
import matplotlib.pyplot as plt

DATA_DIR = '../../../TrainedModels/Stud2'  # Riccardo's folder
DATA_DIR = '../../../TrainedModels/Taught2'  # Riccardo's folder

name = '_student_self_taught'
name = '_student_taught'
units = [8, 16, 32, 64]
datasets = ["DrDrive_DK", "ht1-", "muff-"]
model = 'LSTM_'
DK = 'DK1_'
DK = 'DK2_'

for dataset in datasets:

    for unit in units:
        folder = DK + model+dataset+str(unit) + name

        file = glob.glob(os.path.normpath('/'.join([DATA_DIR, folder, 'WavPredictions', '*_tar.wav'])))[0]
        fs, y = wavfile.read(file)
        file = glob.glob(os.path.normpath('/'.join([DATA_DIR, folder, 'WavPredictions', '*_pred.wav'])))[0]
        fs, predictions = wavfile.read(file)

        y = y[:len(predictions)]
        mse = tf.get_static_value(
            tf.keras.metrics.mean_squared_error(y, predictions))
        mae = tf.get_static_value(
            tf.keras.metrics.mean_absolute_error(y, predictions))
        esr = tf.get_static_value(ESR(y, predictions))
        rmse = tf.get_static_value(RMSE(y, predictions))
        stft = tf.get_static_value(STFT_loss(y, predictions))

        results_ = {'mse': mse, 'mae': mae, 'esr': esr, 'rmse': rmse, 'stft': stft}
        with open(os.path.normpath('/'.join([DATA_DIR, folder, folder + '_results2.txt'])), 'w') as f:
            for key, value in results_.items():
                print('\n', key, '  : ', value, file=f)