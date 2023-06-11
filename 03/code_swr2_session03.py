# Code for seminar SWR2 session 03
from scipy import fft
import soundfile as sf
from matplotlib import pyplot as plt
import numpy as np
import librosa
import librosa.display
from matplotlib import cm
import sounddevice as sd

sample_rate = 44100  # samples per second = Hertz = 1/s

duration = 1.0
times = np.linspace(0.0, duration, num=int(duration * sample_rate))

# 01. Sine wave
# =============

amplitude = 0.8
phase = np.pi / 2
freq = 5.0

signal = amplitude * np.sin(freq * (2 * np.pi) * times + phase)
plt.plot(times, signal, label=f"freq={freq}; phase={phase}")
plt.legend()
plt.show()


amplitude = 0.8
phase = np.pi / 2
freq = 440.0

signal = amplitude * np.sin(freq * (2 * np.pi) * times + phase)
plt.plot(times, signal, label=f"freq={freq}; phase={phase}")
plt.legend()
plt.show()


# 20 millisecond ramp

def ntsf(xx, *, kk=0.0):
    """
    Dino Dini's beautiful normalized tunable sigmoid function.

    https://dhemery.github.io/DHE-Modules/technical/sigmoid/

    """
    return (xx - kk * xx) / (kk - 2 * kk * np.abs(xx) + 1)


ramp = 0.5 * (1.0 + ntsf(np.linspace(-1.0, 1.00,
              num=int(0.02 * sample_rate)), kk=-0.5))

plt.plot(ramp)
plt.title("The ramp.")
plt.show()


signal[:len(ramp)] *= ramp
signal[-len(ramp):] *= ramp[::-1]

plt.plot(times, signal, label=f"freq={freq}; phase={phase}")
plt.title("Signal with 20 ms ramp in and out")
plt.legend()
plt.show()


def apply_ramp(signal, *, ramp):
    """apply a ramp to a copy of the signal at the start and the end of the
    signal"""
    signal = signal.copy()
    signal[:len(ramp)] *= ramp
    signal[-len(ramp):] *= ramp[::-1]
    return signal


sd.play(signal, samplerate=sample_rate)


# 02. Tones (mixing sine waves)
# =============================

amplitude = 0.8
phase = 0.0
freq = 440.0

a440 = amplitude * np.sin(freq * (2 * np.pi) * times + phase)

freq = 294
d294 = amplitude * np.sin(freq * (2 * np.pi) * times + phase)

freq = 349
f349 = amplitude * np.sin(freq * (2 * np.pi) * times + phase)

freq = 523
c523 = amplitude * np.sin(freq * (2 * np.pi) * times + phase)


signal = apply_ramp(f349, ramp=ramp)
sd.play(signal, samplerate=sample_rate)


# now start mixing, but we want to keep the maximal amplitude in check.

tone1 = (d294 + a440) / 2

signal = apply_ramp(tone1, ramp=ramp)
sd.play(signal, samplerate=sample_rate)


tone2 = (d294 + a440 + f349) / 3

signal = apply_ramp(tone2, ramp=ramp)
sd.play(signal, samplerate=sample_rate)


tone3 = (d294 + a440 + f349 + c523) / 4

signal = apply_ramp(tone3, ramp=ramp)
sd.play(signal, samplerate=sample_rate)

plt.plot(times, signal, label=f"tone3")
plt.title("full signal")
plt.legend()
plt.show()

# one last tone with different amplitudes in the source sinus waves
tone4 = 0.5 * d294 + 0.3 * a440 + 0.15 * f349 + 0.05 * c523
signal = apply_ramp(tone3, ramp=ramp)
sd.play(signal, samplerate=sample_rate)

plt.plot(times, tone4, label=f"tone4")
# plt.plot(times, tone3, label=f"tone3")
# plt.plot(times, tone2, label=f"tone2")
plt.plot(times, tone1, label=f"tone1")
plt.plot(times, d294, label=f"d294")
plt.title("zoomed in")
plt.xlim(0.10, 0.12)
plt.legend()
plt.show()


# 03. Decomposing the tone
# ========================


tone3_fft = fft.fft(tone3)
tone3_recov = fft.ifft(tone3_fft)

plt.plot(times, tone3, label="tone3")
plt.plot(times, tone3_recov.real, label="tone3 recovered")
plt.legend()
plt.show()

# the orange line is ontop of the blue line

plt.plot(tone3_fft.real, label='tone3 freq real')
plt.plot(tone3_fft.imag, label='tone3 freq imag')
plt.legend()
plt.show()


plt.plot(tone3_fft.real, label='tone3 freq real')
plt.plot(tone3_fft.imag, label='tone3 freq imag')
plt.xlim(0, 600)
plt.ylim(-200, 200)
plt.title('Zoomed in')
plt.legend()
plt.show()

# reduce fft values to relevant part
# half of the values are hermitian symmetric / the complex conjugate of the
# first part

tone3_fft_truncated = tone3_fft[:len(tone3_fft)//2]

# the frequencies corresponding to each value can be calculated the following
window_length = len(tone3)
freqs = np.arange(len(tone3)//2) * sample_rate / window_length

plt.plot(freqs, tone3_fft_truncated.real, label='real')
plt.plot(freqs, tone3_fft_truncated.imag, label='imag')
plt.xlabel("Frequency in [Hz]")
plt.legend()
plt.show()

tone3_magnitude = (tone3_fft_truncated * tone3_fft_truncated.conj()).real
plt.plot(tone3_fft.real, label='real')
plt.plot(tone3_fft.imag, label='imag')
plt.plot(tone3_magnitude, label='magnitude')
plt.xlim(0, 600)
plt.ylim(-200, 200)
plt.title('Zoomed in')
plt.legend()
plt.show()


# find the peaks in the magnitude


plt.plot(tone3_magnitude, label='magnitude')
plt.xlim(0, 600)
plt.show()

sum(tone3_magnitude > 1.80e7)

# There are four values highter than 18,000,000

np.where(tone3_magnitude > 1.80e7)

# They are at positoins [294, 349, 440, 523]
# and correspond to frequencies:

freqs[np.where(tone3_magnitude > 1.80e7)]

# [294., 349., 440., 523.]


# Assignment:

# 1. Decompose tone2, what are the frequencies you find?


# 2. Read in balmer8.flac and decompose the signal. Which are the frequencies used?

# TODO

# 3. Read in lyman6.flac and decompose the signal. Which are the frequencies
# used? Bonus: With which amplitude are they mixed together?

# TODO
