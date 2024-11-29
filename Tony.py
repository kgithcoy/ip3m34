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

def wavfile_plot(x,Fs,name,tinterest,freqinterest):
    """
    Plots the time-domain and frequency-domain representation of an audio signal.

    Parameters:
    - x: The audio data (numpy array).
    - Fs: Sampling frequency in Hz (float).
    - name: Name of the signal (string).
    - tinterest: time-domain  x-axis(time) and y-axis(amplitude) limiting zoom in ([[xlim],[ylim]]). 0 for whole axis
    - freqinterest: frequency-domain x-axis(frequencu) and y-axis(magnitude) limiting zoom in ([[xlim],[ylim]]). 0 for whole axis
    """
    t = np.linspace(0, len(x)/Fs, len(x))
    f = np.fft.fftfreq(len(x), d=1/Fs)
    X = np.fft.fft(x)
    X_absolute = np.abs(X)
    fig, ax = plt.subplots(1, 2, figsize=(20, 5))
    ax[0].plot(t, x)
    ax[0].set_title(f"Time domain: {name}")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time [s]")
    if tinterest != 0:
        ax[0].set_xlim(tinterest[0])
        ax[0].set_ylim(tinterest[1])
    ax[0].grid()
    ax[1].plot(f[:len(f) // 2], X_absolute[:len(f) // 2])
    ax[1].set_title(f"Frequency domain: {name}")
    ax[1].set_ylabel("Magnitude")
    ax[1].set_xlabel("Frequency [Hz]")
    if freqinterest != 0:
        ax[1].set_xlim(freqinterest[0])
        ax[1].set_ylim(freqinterest[1])
    ax[1].grid()
    fig.tight_layout()
    plt.show()

def response_plot(x,Fs,name,logfreq):
    """
    Plots the time-domain and frequency-domain representation of an response signal

    Parameters:
    - x: The response data (numpy array).
    - Fs: Sampling frequency in Hz (float).
    - name: Name of the signal (string).
    - logfreq: Boolean indication if logarithic x-axis is whished. (binary)
    """
    t = np.linspace(0, len(x)/Fs, len(x))
    f = np.linspace(0, Fs, len(x))
    X = np.fft.fft(x)
    X_absolute = np.abs(X)
    X_phase = np.angle(X)
    unwrapped_phase = np.unwrap(X_phase)
    fig, ax = plt.subplots(3, 1, figsize=(20, 10))
    ax[0].plot(t, x)
    ax[0].set_title(f"Time domain: {name}")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time [s]")
    ax[0].grid()
    ax[1].plot(f, 20 * np.log10(X_absolute + 1e-15))
    ax[1].set_title(f"Frequency domain: {name}")
    ax[1].set_ylabel("Magnitude [dB]")
    ax[1].set_xlabel("Frequency [Hz]")
    if logfreq == 1:
        ax[1].set_xscale("log")
    ax[1].grid()
    ax[2].plot(f, X_phase)
    ax[2].set_title(f"Frequency domain: {name}")
    ax[2].set_ylabel("Phase [rad]")
    ax[2].set_xlabel("Frequency [Hz]")
    if logfreq == 1:
        ax[2].set_xscale("log")
    ax[2].grid()
    fig.tight_layout()
    plt.show()

def give_M(Fs_new, Fs_old):
    M = Fs_old // Fs_new # Downsampling factor
    return M

def wavfile_plot_combined(x1, x2, Fs, name1, name2):
    """
    Plots the time-domain and frequency-domain representation of two audio signals in one figure.

    Parameters:
    - x1: The first audio data (numpy array).
    - x2: The second audio data (numpy array).
    - Fs: Sampling frequency in Hz (float).
    - name1: Name of the first signal (string).
    - name2: Name of the second signal (string).
    """
    # Time and frequency axes
    t = np.linspace(0, len(x1)/Fs, len(x1))
    f = np.fft.fftfreq(len(x1), d=1/Fs)
    # FFT results for both signals
    X1 = np.fft.fft(x1)
    X2 = np.fft.fft(x2)
    X1_absolute = np.abs(X1)
    X2_absolute = np.abs(X2)
    # Create subplots
    fig, ax = plt.subplots(1, 3, figsize=(20, 5))
    # Time-domain plot
    ax[0].plot(t, x1, label=name1, alpha=0.7)
    ax[0].plot(t, x2, label=name2, alpha=0.7)
    ax[0].set_title("Time Domain")
    ax[0].set_ylabel("Amplitude")
    ax[0].set_xlabel("Time [s]")
    ax[0].legend()
    ax[0].grid()
    # Frequency-domain plot for first signal
    ax[1].plot(f[:len(f)//2], X1_absolute[:len(f)//2], label=f"{name1} Frequency", color="blue")
    ax[1].set_title(f"Frequency Domain: {name1}")
    ax[1].set_ylabel("Magnitude")
    ax[1].set_xlabel("Frequency [Hz]")
    ax[1].grid()
    # Frequency-domain plot for second signal
    ax[2].plot(f[:len(f)//2], X2_absolute[:len(f)//2], label=f"{name2} Frequency", color="orange")
    ax[2].set_title(f"Frequency Domain: {name2}")
    ax[2].set_ylabel("Magnitude")
    ax[2].set_xlabel("Frequency [Hz]")
    ax[2].grid()
    fig.tight_layout()
    plt.show()

def downsampling(x, Fs_x, Fs_new):
    """
    Downsample a signal to a new sampling rate 

    Parameters:
    - x (array-like): Input signal to be downsampled.
    - Fs_x (float): Original sampling rate of the signal.
    - Fs_new (float): Desired new sampling rate.

    Returns:
    - downsampled_signal (array-like): Signal downsampled to the new sampling rate.
    """
    if Fs_new >= Fs_x:
        raise ValueError(f"New sampling rate ({Fs_new}) must be less than the original rate ({Fs_x}).")
    M = give_M(Fs_new, Fs_x)
    downsampled_signal = x[::M]
    return downsampled_signal

def normalize(x):
    x = x / np.max(np.abs(x))
    return x

def shannon_energy(x):
    x = np.abs(x)
    x_squared = x**2
    E_s = ( -x_squared ) * np.log10(np.maximum(x_squared,1e-15))
    return E_s

def zero_phase_butter_filter(x, Fs, order, cutoff, filter_type):
    """
    Zero-phase Butterworth filtering with filtfilt.

    Parameters:
    - x: Input signal (numpy array).
    - Fs: Sampling frequency in Hz (float).
    - order: Filter order (int).
    - cutoff: Cutoff frequency in Hz (float) for lowpass/highpass or list of two floats for bandpass/bandstop.
    - filter_type: Type of filter ('low', 'high', 'band', 'stop').

    Returns:
    - filtered_x: Filtered signal (numpy array).
    """
    if isinstance(cutoff, (list, tuple)) and len(cutoff) == 2:
        # Bandpass or bandstop filter
        low_freq = cutoff[0]
        high_freq = cutoff[1]
        low = low_freq / (Fs / 2)
        high = high_freq / (Fs / 2)
        b, a = butter(order, [low, high], btype=filter_type)
    else:
        # Lowpass or highpass filter
        cutoff_norm = cutoff / (Fs / 2)
        b, a = butter(order, cutoff_norm, btype=filter_type)
    
    # Zero-phase filtering
    filtered_x = filtfilt(b, a, x)
    return filtered_x


def check_attenuation(b, a, Fs, F_interest, worn=8000):
    """
    Check the dB attenuation of a filter at a specified frequency.

    Parameters:
    - b, a: Filter coefficients (numerator and denominator).
    - Fs: Sampling frequency in Hz.
    - F_interest: Frequency of interest to check the attenuation (in Hz).
    - worn: Number of frequency points to compute the frequency response (default=8000).

    Returns:
    - dB_value: The dB attenuation at the frequency of interest.
    """
    w, h = freqz(b, a, worN=worn)   # Compute frequency response
    f = (Fs / 2) * w  / np.pi       # Convert frequency axis to Hz depending on the sampling rate Fs (y-axis)
    dB_response = 20 * np.log10(abs(h)+1e-15) # Compute the dB-response (x-axis)
    index = np.argmin(np.abs(f - F_interest))
    dB_value = dB_response[index]
    print(f"The dB attenuation at {F_interest} Hz is {dB_value:.2f} dB.")
    return dB_value

def spectogram(x, Fs, t_interest, freqinterest, nperseg=256, noverlap=128):
    """
    Plot the spectogram of a given signal.

    Parameters:
    - x: Input of the analysed signal (numpy array).
    - Fs: Sampling rate of the signal in Hz (float).
    - t_interest: The time-interval of interest (array of two indices), put 0 if want whole axis.
    - freqinterest: The frequency-interval of interest, note is double-sided, put 0 if want whole axis.





    """
    window = gaussian(nperseg, std=1e-2 * Fs) # Make a Gaussian windo
    freq, times, Sxx = spectrogram(           # Compute the spectogram variables
    x,
    fs=Fs,
    window=window,
    nperseg=nperseg,
    noverlap=noverlap,
    nfft=nperseg,
    detrend='constant',
    return_onesided=False,  # Return both positive and negative frequencies
    scaling='density',
    mode='psd')  # Power Spectral Density
    Sxx = np.fft.fftshift(Sxx, axes=0)
    Sx_dB = 10 * np.log10(np.maximum(Sxx, 1e-15))
    plt.figure(figsize=(12, 6))
    plt.imshow(
        Sx_dB,
        origin="lower",
        aspect="auto",
        extent=[times.min(), times.max(), freq.min(), freq.max()],
        cmap="viridis"
    )
    plt.colorbar(label="Power Spectral Density (dB)")
    plt.title("Spectogram")
    plt.xlabel("Time [s]")
    plt.ylabel("Frequency [Hz]")
    if t_interest != 0:
        plt.xlim(t_interest)
    if freqinterest != 0:
        plt.ylim(freqinterest)
    plt.show()
    


    