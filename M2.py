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
from scipy.signal import find_peaks, peak_prominences, peak_widths
from scipy.signal import stft, TransferFunction, lsim

# Import own made utilies:
from Tony import *


# Import audiorecording
#audio_sample_rate, data= wavfile.read("50763_AV kopie.wav")
#audio_sample_rate, data= wavfile.read("heart_single_channel_physionet_49829_TV kopie.wav")
audio_sample_rate, data = wavfile.read("recording_heart_ownDevice kopie.wav")
data = data.astype(float)
#data = normalize(data) # If needed, normalize the data

# Display details
print(f"Audio Sample Rate: {audio_sample_rate} Hz")
print(f"Number of Samples: {len(data)}")
print(f"Shape of the recording: {data.shape}")

#---------------------------------------------------------------------------------------------------------------------------------
# M1:

# Parameters:
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
# Segmentation
#segmentation(SEE, Fs_new, 0.4, verbose=False)

# Plots:
#spectogram(downsampled_data_normalize, Fs_new, 0, 0)
#wavfile_plot(shannon_signal, Fs_new, "Shannon Energy",[[0,10],0],0)
#wavfile_plot(SEE, Fs_new, "SEE", [[2,8],0], 0)
#wavfile_plot(filtered_data, Fs_new, "Band Filterd signal 2kHz of heart_single_channel_physionet_49829_TV", 0, 0)
#wavfile_plot_combined(shannon_signal, SEE, Fs_new, "Shannon energy", "SEE")
#wavfile_plot(downsampled_data, Fs_new, "Downsampled heart_single_channel_physionet_49829_TV",0,0)
#wavfile_plot(data, audio_sample_rate, "heart_single_channel_physionet_49829_TV",0,0)
#wavfile_plot(downsampled_data_normalize, Fs_new, "Downsampled Normalized heart_single_channel_physionet_49829_TV",0,0)
#wavfile_plot_combined(downsampled_data, downsampled_data_normalize, Fs_new, "Downsampled unnormalized", "Downsampled normalized")

#---------------------------------------------------------------------------------------------------------------------------------
# M2:

# Parameters:
Fs = 4000
T_total = 1 # in seconds
v_sound = 60  # Speed of sound in the body in m/s
repetitions = 10 # Repete the heartbeats response this many time

# Computing all valves response M, T, A, P (with or without noise)
M_t, M = pole_pairs(20, 50, 1, 10, Fs, T_total, 10, addnosie=False)
T_t, T = pole_pairs(20, 150, 0.5, 40, Fs, T_total, 10, addnosie=False)
A_t, A = pole_pairs(20, 50, 0.5, 300, Fs, T_total, 10, addnosie=False)
P_t, P = pole_pairs(20, 30, 0.4, 330, Fs, T_total, 10, addnosie=False)
# System of multichannel:
# Valve, location of [x,y,z] in cm!
valve = np.array([[6.37, 10.65, 6.00],     # Valve[0] = M , The assigned position index of the valve name is important
                    [0.94, 9.57, 5.50],    # Valve[1] = T
                    [5.50, 11.00, 3.60],   # Valve[2] = A
                    [3.90, 11.50, 4.50]])  # Valve[3] = P

# Microphone, location of [x,y,z] in cm!
microphone = np.array([[2.5, 5.0, 0],    # Mic[0]
                        [2.5, 10.0, 0],  # Mic[1]
                        [2.5, 15.0, 0],  # Mic[2]
                        [7.5, 5.0, 0],   # Mic[3] 
                        [7.5, 10.0, 0],  # Mic[4]
                        [7.5, 15.0, 0]]) # Mic[5]

# Compute the unfiltered multichannel, if needed we can plot it all in one figure
microphone_multichannel_unfiltered, distance, attenuation, delay = multi_channel_signal(M, T, A, P, Fs, valve, microphone, plotting=True)

# Display details
print(f"Micrphone output:{microphone_multichannel_unfiltered}")
print(f"Distance:{distance}")
print(f"Attenuation:{attenuation}")
print(f"Delay:{delay}")

# Plots:
heartbeat_plot(M ,T, A, P, M_t, T_t, A_t ,P_t, Fs, T_total, repetitions=repetitions)







