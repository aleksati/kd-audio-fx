
from scipy.io import wavfile
from librosa import display
import matplotlib.pyplot as plt

fs2, audio2 = wavfile.read(
    "..\\TrainedModels\\LSTMCL1B_DK8_student_taught\\WavPredictions\\0_pred.wav")

fs, audio = wavfile.read(
    "..\\TrainedModels\\LSTMCL1B_DK8_student_self_taught\\WavPredictions\\0_pred.wav")

fig, ax = plt.subplots(nrows=1, ncols=1)
display.waveshow(audio, sr=fs, ax=ax)
display.waveshow(audio2, sr=fs2, ax=ax)
plt.show()
