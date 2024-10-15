import numpy as np
from scipy.io import wavfile
import os
import glob
import pickle
import matplotlib.pyplot as plt
from Utils import filterAudio


DATA_DIR = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/DK/DrDriveCond_DK'
SAVE_DIR = 'C:/Users/riccarsi/OneDrive - Universitetet i Oslo/Datasets/DK/'

def data_preparation_DRIVECond():#

    start = 0
    lim = 22250000

    tars = []
    inps = []
    tars_test = []
    inps_test = []

    file = glob.glob(os.path.normpath('/'.join([DATA_DIR, 'input', '*.wav'])))[0]
    fs, audio = wavfile.read(file)

    inps.append(audio[start:lim])
    inps_test.append(audio[lim:])

    files = glob.glob(os.path.normpath('/'.join([DATA_DIR, '*.wav'])))
    conds = []
    for file in files:
        fs, audio = wavfile.read(file)

        filename = os.path.split(file)[-1]
        filename = filename[:-4]
        param = filename.split('-')[1]
        conds.append(float(param.split('_')[0]))

        tars.append(audio[start:lim])
        tars_test.append(audio[lim:])

    inps = np.array(inps, dtype=np.float32)
    tars = np.array(tars, dtype=np.float32)
    conds = np.array(conds, dtype=np.float32).reshape(-1, 1)

    data = {'z': conds, 'y': tars, 'x': inps}

    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'DrDriveCond_DK_train.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

    inps = np.array(inps_test, dtype=np.float32)
    tars = np.array(tars_test, dtype=np.float32)

    data = {'z': conds, 'y': tars, 'x': inps}
    #
    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'DrDriveCond_DK_test.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

def data_preparation_DRIVE():#

    start = 0
    lim = 22250000

    tars = []
    inps = []
    tars_test = []
    inps_test = []

    file = glob.glob(os.path.normpath('/'.join([DATA_DIR, 'DrDrive_input.wav'])))[0]
    fs, audio = wavfile.read(file)

    inps.append(audio[start:lim])
    inps_test.append(audio[lim:])

    file = glob.glob(os.path.normpath('/'.join([DATA_DIR, 'DrDrive_target.wav'])))[0]
    fs, audio = wavfile.read(file)

    tars.append(audio[start:lim])
    tars_test.append(audio[lim:])

    inps = np.array(inps, dtype=np.float32)
    tars = np.array(tars, dtype=np.float32)

    data = {'z': None, 'y': tars, 'x': inps}

    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'DrDrive_DK_train.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

    inps = np.array(inps_test, dtype=np.float32)[:, :lim]
    tars = np.array(tars_test, dtype=np.float32)

    data = {'z': None, 'y': tars, 'x': inps}
    #
    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'DrDrive_DK_test.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

def data_preparation_CL1B():#



    start = 0
    lim = 22250000


    min_ = 1e9

    tars = []
    inps = []
    tars_test = []
    inps_test = []

    file = glob.glob(os.path.normpath('/'.join([DATA_DIR, 'CL1B_input.wav'])))[0]
    fs, audio = wavfile.read(file)
    #audio = Code.audio_format.pcm2float(audio)

    # plt.plot(audio)
    inps.append(audio[start:lim])
    inps_test.append(audio[lim:])

    file = glob.glob(os.path.normpath('/'.join([DATA_DIR, 'CL1B_target.wav'])))[0]
    fs, audio = wavfile.read(file)

    tars.append(audio[start:lim])
    tars_test.append(audio[lim:])

    inps = np.array(inps, dtype=np.float32)
    tars = np.array(tars, dtype=np.float32)

    data = {'z': None, 'y': tars, 'x': inps}

    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'CL1B_DK_train.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()

    inps = np.array(inps_test, dtype=np.float32)[:, :lim]
    tars = np.array(tars_test, dtype=np.float32)

    data = {'z': None, 'y': tars, 'x': inps}
    #
    file_data = open(os.path.normpath('/'.join([SAVE_DIR, 'CL1B_DK_test.pickle'])), 'wb')
    pickle.dump(data, file_data)
    file_data.close()


if __name__ == '__main__':

    #data_preparation_CL1B()
    #data_preparation_DRIVE()
    data_preparation_DRIVECond()