import numpy as np                  #Always include this
import matplotlib.pyplot as plt     #For plotting
from scipy.io import wavfile        #For using .wav type datas
from IPython.display import Audio   #For audio display if needed

# Scipy modules
from scipy.signal import stft
from scipy import signal
#from scipy.fft import fft, ifft    #If we have this, then np.fft.abcd in codes can be changed to abcd
from scipy.signal import butter, filtfilt
from scipy.signal import convolve, unit_impulse, freqz, spectrogram
from scipy.signal.windows import gaussian

# Import own made utilies:
from Tony import *

# Import audiorecording
audio_sample_rate, data = wavfile.read("heart_single_channel_physionet_49829_TV.wav")
data = data.astype(float)
#data = normalize(data)

# Display details
print(f"Audio Sample Rate: {audio_sample_rate} Hz")
print(f"Number of Samples: {len(data)}")
print(f"Shape of the recording: {data.shape}")

Fs_new = 3998 #[Hz]
cutoff_low = 20 # Filtering out DC-components
cutoff_high = 200
order = 2

# Checking for sufficient attenuation if needed
b, a = butter(order, [cutoff_low/(audio_sample_rate/2),cutoff_high/(audio_sample_rate/2)], btype="band")
check_attenuation(b, a, audio_sample_rate, audio_sample_rate/2 )
# Filtering the original data
filtered_data = zero_phase_butter_filter(data, audio_sample_rate, order, [cutoff_low, cutoff_high], "band")
# Downsampling
M = give_M(Fs_new, audio_sample_rate)
print(f"M = {M}")
downsampled_data = downsampling(filtered_data, audio_sample_rate, Fs_new)
downsampled_data_normalize = normalize(downsampled_data) #Normalizig the downsampled signal
# Compute the SEE
shannon_signal = shannon_energy(downsampled_data_normalize)
SEE = zero_phase_butter_filter(shannon_signal, Fs_new, order, 15, "low") # Apply lowpass-(zero-phase)filtering at 15Hz
SEE = normalize(SEE)

# Plots:
spectogram(downsampled_data_normalize, Fs_new, 0, 0)
wavfile_plot(shannon_signal, Fs_new, "Shannon Energy",[[0,10],0],0)
wavfile_plot(SEE, Fs_new, "SEE", [[2,8],0], 0)
wavfile_plot(filtered_data, Fs_new, "Band Filterd signal 2kHz of heart_single_channel_physionet_49829_TV", 0, 0)
wavfile_plot_combined(shannon_signal, SEE, Fs_new, "Shannon energy", "SEE")
wavfile_plot(downsampled_data, Fs_new, "Downsampled heart_single_channel_physionet_49829_TV",0,0)
#wavfile_plot(data, audio_sample_rate, "heart_single_channel_physionet_49829_TV",0,0)
#wavfile_plot(downsampled_data_normalize, Fs_new, "Downsampled Normalized heart_single_channel_physionet_49829_TV",0,0)
#wavfile_plot_combined(downsampled_data, downsampled_data_normalize, Fs_new, "Downsampled unnormalized", "Downsampled normalized")






