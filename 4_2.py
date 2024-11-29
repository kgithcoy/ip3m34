import numpy as np
from scipy.signal import TransferFunction, step, impulse, freqresp
import matplotlib.pyplot as plt
from scipy import signal
from scipy.io import wavfile
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.signal import convolve, unit_impulse, freqz, spectrogram
from scipy.signal import ShortTimeFFT
from scipy.signal.windows import gaussian
from Tony import *

Fs = 4000
T_total = 1
v_sound = 60  # Speed of sound in the body in m/s

_, M = pole_pairs(20, 50, 1, 10, Fs, T_total)
_, T = pole_pairs(20, 150, 0.5, 40, Fs, T_total)
_, A = pole_pairs(20, 50, 0.5, 300, Fs, T_total)
_, P = pole_pairs(20, 30, 0.4, 330, Fs, T_total)

combined_response = M + T + A + P

#normalize
combined_response_normalize = combined_response / np.max(np.abs(combined_response))

#repeat the response 10 times
repetitions = 10
repeated_response = np.tile(combined_response_normalize, repetitions)

# Plots
plt.plot(np.linspace(0, T_total * repetitions, int(Fs * T_total * repetitions)), repeated_response)
plt.title('Repeated Impulse Response of All Valves (10 times)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.show()

sf.write('heartbeat_2.wav', repeated_response, Fs)
print(os.path.abspath('heartbeat_repeated.wav'))

