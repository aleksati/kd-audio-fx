import numpy as np
from scipy.io import wavfile
import os
import glob
import pickle
import Code.audio_format
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from Utils import filterAudio


DATA_DIR = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/Compressors/CL1B_n/Audio'
DATA_DIR = '../../Files'
SAVE_DIR = '../../Files'
SAVE_DIR = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/Compressors/Pickles'
SAVE_DIR = '../../Files'

def data_preparation_CL1B(folder):#



    start = 576000
    lim = int(134.45*48000)
    end = 2918644#2900000#2918644

    min_ = 1e9

    tars = []
    inps = []
    tars_test = []
    inps_test = []

    file = glob.glob(os.path.normpath('/'.join([DATA_DIR, 'input.wav'])))[0]
    fs, audio = wavfile.read(file)
    audio = Code.audio_format.pcm2float(audio)

    # plt.plot(audio)
    inps.append(audio[start:lim])
    inps_test.append(audio[lim:lim+end])
    peaks_ = find_peaks(audio[:3000], height=0.1)[0][0]

    file_dirs = glob.glob(os.path.normpath('/'.join([DATA_DIR, 'target.wav'])))

    audio = Code.audio_format.pcm2float(audio)


    peaks = find_peaks(audio[:3000], height=0.01)[0][0]
    if peaks_ != peaks:
        audio = audio[np.abs(peaks_-peaks):]
        #plt.plot(audio[:3000])
        #plt.plot(inps[0][:3000])


    if min_ > len(audio[lim:]):
        min_ = len(audio[lim:])

    tars.append(audio[start:lim])
    tars_test.append(audio[lim:lim+end])

    inps = np.array(inps, dtype=np.float32)
    tars = np.array(tars, dtype=np.float32)

    data = {'z': None, 'y': tars, 'x': inps}

    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'CL1B_DK_train' + folder + '.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

    inps = np.array(inps_test, dtype=np.float32)[:, :lim+end]
    tars = np.array(tars_test, dtype=np.float32)

    data = {'z': None, 'y': tars, 'x': inps}
    #
    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'CL1B_DK_test' + folder + '.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()


if __name__ == '__main__':



    data_preparation_CL1B_analog()

