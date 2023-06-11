
import soundfile as sf
import numpy as np
import sounddevice as sd


sample_rate = 44100  # samples per second = Hertz = 1/s

duration = 1.5
times = np.linspace(0.0, duration, num=int(duration * sample_rate))

amplitude = 0.8
phase = 0.0

# values in nm used as Hz rounded to integer

# lyman
l2 = amplitude * np.sin(122 * (2 * np.pi) * times + phase)
l3 = amplitude * np.sin(103 * (2 * np.pi) * times + phase)
l4 = amplitude * np.sin(97 * (2 * np.pi) * times + phase)
l5 = amplitude * np.sin(95 * (2 * np.pi) * times + phase)
l6 = amplitude * np.sin(94 * (2 * np.pi) * times + phase)

signal = 0.25 * l2 + 0.1111111111111111 * l3 + \
    0.0625 * l4 + 0.04 * l5 + 0.027777777777777776 * l6

# re-normalize
signal *= 1 / sum(1 / n ** 2 for n in range(2, 7))


def ntsf(xx, *, kk=0.0):
    """
    Dino Dini's beautiful normalized tunable sigmoid function.

    https://dhemery.github.io/DHE-Modules/technical/sigmoid/

    """
    return (xx - kk * xx) / (kk - 2 * kk * np.abs(xx) + 1)


def apply_ramp(signal, *, ramp):
    """apply a ramp to a copy of the signal at the start and the end of the
    signal"""
    signal = signal.copy()
    signal[:len(ramp)] *= ramp
    signal[-len(ramp):] *= ramp[::-1]
    return signal


ramp = 0.5 * (1.0 + ntsf(np.linspace(-1.0, 1.00,
              num=int(0.02 * sample_rate)), kk=-0.5))

signal = apply_ramp(signal, ramp=ramp)

sd.play(signal, samplerate=sample_rate)

sf.write('lyman6.flac', signal, sample_rate)


# balmer
duration = 1.0
times = np.linspace(0.0, duration, num=int(duration * sample_rate))

b3 = amplitude * np.sin(656 * (2 * np.pi) * times + phase)
b4 = amplitude * np.sin(486 * (2 * np.pi) * times + phase)
b5 = amplitude * np.sin(434 * (2 * np.pi) * times + phase)
b6 = amplitude * np.sin(410 * (2 * np.pi) * times + phase)
b7 = amplitude * np.sin(397 * (2 * np.pi) * times + phase)
b8 = amplitude * np.sin(388 * (2 * np.pi) * times + phase)

signal = b3 + b4 + b5 + b6 + b7 + b8

signal *= 1 / 6  # re-normalize
signal = apply_ramp(signal, ramp=ramp)
sd.play(signal, samplerate=sample_rate)
sf.write('balmer8.flac', signal, sample_rate)
