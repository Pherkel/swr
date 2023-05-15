import soundfile as sf
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
import librosa

wav, sample_rate = sf.read('beispiel.wav')
print(wav.shape)
print(sample_rate)

#plt.plot(wav)
#plt.show()

melspec = librosa.feature.melspectrogram(y=wav[:, 0], n_fft=1024, hop_length=220, n_mels=60, sr=sample_rate,  power=1.0, fmin=10, fmax=12000)
melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)

librosa.display.specshow(melspec_db, y_axis='mel',  x_axis='time', sr=44100, hop_length=220, cmap=cm.magma)
plt.show()

def librosa_melspec(wav, sample_rate):
    wav = librosa.resample(wav, orig_sr=sample_rate, target_sr=44100,
            res_type='kaiser_best', fix=True, scale=False)
    melspec = librosa.feature.melspectrogram(y=wav, n_fft=1024, hop_length=220, n_mels=60, sr=44100, power=1.0, fmin=10, fmax=12000)
    melspec_db = librosa.amplitude_to_db(melspec, ref=0.15)
    return np.array(melspec_db.T, order='C', dtype=np.float64)