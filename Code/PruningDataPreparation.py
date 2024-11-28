import numpy as np
from scipy.io import wavfile
import os
import glob
import pickle
import Code.audio_format
import matplotlib.pyplot as plt


DATA_DIR = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/DK/PrunedGuitarVA-master/Data'
SAVE_DIR = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/DK/PrunedGuitarVA-master/Data'


def data_preparation():
    devices = ['ht1-', 'muff-']
    folders = ['train', 'val', 'test']

    for device in devices:
        for folder in folders:

            dir = os.path.normpath('/'.join([DATA_DIR, folder]))
            file = glob.glob(os.path.normpath('/'.join([dir, device + 'input.wav'])))[0]
            fs, audio = wavfile.read(file)
            inp = Code.audio_format.pcm2float(audio)

            file = glob.glob(os.path.normpath('/'.join([dir, device + 'target.wav'])))[0]
            fs, audio = wavfile.read(file)
            tar = Code.audio_format.pcm2float(audio)
        #
    #
            inps = np.array(inp, dtype=np.float32).reshape(1, -1)
            tars = np.array(tar, dtype=np.float32).reshape(1, -1)

            data = {'y': tars, 'x': inps}

            file_data = open(os.path.normpath('/'.join([SAVE_DIR, device + '_' + folder + '.pickle'])), 'wb')
            pickle.dump(data, file_data)
            file_data.close()



if __name__ == '__main__':

    #####already done
    data_preparation()