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

def wavfile_plot(x,Fs,name,tinterest,freqinterest):
    """
    Plots the time-domain and frequency-domain representation of an audio signal.

    Parameters:
    - x: The audio data (array-like).
    - Fs: Sampling frequency in Hz (float).
    - name: Name of the signal (string).
    - tinterest: time-domain  x-axis(time) and y-axis(amplitude) limiting zoom in ([[xlim],[ylim]]). 0 for whole axis.
    - freqinterest: frequency-domain x-axis(frequencu) and y-axis(magnitude) limiting zoom in ([[xlim],[ylim]]). 0 for whole axis.
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

def response_plot(x,Fs,name,logfreq=False):
    """
    Plots the time-domain and frequency-domain representation of an response signal.

    Parameters:
    - x: The response data (array-like).
    - Fs: Sampling frequency in Hz (float).
    - name: Name of the signal (string).
    - logfreq: Boolean indication if logarithic x-axis is whished, default=False (Boolean).
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
    ax[1].grid()
    ax[2].plot(f, X_phase)
    ax[2].set_title(f"Frequency domain: {name}")
    ax[2].set_ylabel("Phase [rad]")
    ax[2].set_xlabel("Frequency [Hz]")
    ax[2].grid()
    if logfreq:
        ax[1].set_xscale("log")
        ax[2].set_xscale("log")
    fig.tight_layout()
    plt.show()

def give_M(Fs_new, Fs_old):     # Fs_old is the original sampling frequency and Fs_new is the sampling frequency you wanted.
    M = Fs_old // Fs_new        # Compute the Downsampling factor and return it
    return M

def wavfile_plot_combined(x1, x2, Fs, name1, name2):
    """
    Plots the time-domain representation of two audio signals in one figure and plots the the frequency-domain representation independently.

    Parameters:
    - x1: The first audio data (array-like).
    - x2: The second audio data (array-like).
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
    Downsample a signal to a new sampling rate .

    Parameters:
    - x: Input signal to be downsampled (array-like).
    - Fs_x: Original sampling rate of the signal in Hz (float).
    - Fs_new: Desired new sampling rate in Hz ((float)).

    Returns:
    - downsampled_signal: Signal downsampled to the new sampling rate (array-like).
    """
    if Fs_new >= Fs_x:
        raise ValueError(f"New sampling rate ({Fs_new}) must be less than the original rate ({Fs_x}).")
    M = give_M(Fs_new, Fs_x)
    downsampled_signal = x[::M]
    return downsampled_signal

def normalize(x): # Funtion to normalize a list/np.array x.
    x = x / np.max(np.abs(x))
    return x

def shannon_energy(x):
    """
    Compute the shannon-energy of a signal.

    Parameters:
    - x: Input signal (array-like).

    Returns:
    - E_s: The shannon-energy of a signal (array-like).
    """
    x = np.abs(x)
    x_squared = x**2
    E_s = ( -x_squared ) * np.log10(np.maximum(x_squared,1e-15))
    return E_s

def zero_phase_butter_filter(x, Fs, order, cutoff, filter_type):
    """
    Zero-phase Butterworth filtering with filtfilt.

    Parameters:
    - x: Input signal (array-like).
    - Fs: Sampling frequency in Hz (float).
    - order: Filter order (int).
    - cutoff: Cutoff frequency in Hz (float) for lowpass/highpass or list of two floats for bandpass/bandstop.
    - filter_type: Type of filter ('low', 'high', 'band', 'stop').

    Returns:
    - filtered_x: Filtered signal (array-like).
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
    - Fs: Sampling frequency in Hz (float).
    - F_interest: Frequency of interest to check the attenuation in Hz (float).
    - worn: Number of frequency points to compute the frequency response (default=8000).

    Returns:
    - dB_value: The dB attenuation at the frequency of interest (float).
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
    - x: Input of the analysed signal (array-like).
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
    

def pole_pairs(duration_ms, freq_hz, amplitude, onset_delay_ms, Fs, T_total, Ns, addnosie=False):
    
    """
    Simulates the impulse response of a second-order system defined by the given parameters or in other words: generation of the valve-response.

    Parameters:
    - duration_ms: Duration of the impulse signal in milliseconds (float).
    - freq_hz: Natural frequency of the system in Hz (float).
    - amplitude: Amplitude scaling factor for the system's output (float).
    - onset_delay_ms: Delay before the impulse signal starts, in milliseconds (float).
    - Fs: Sampling frequency in Hz (float).
    - T_total: Total duration of the simulation in seconds (float).
    - Nrep: Number of non-zero samples to include in the random noise if `addnoise` is True (int).
    - addnoise: If True, adds random noise to the impulse signal; otherwise, a single impulse is use, default=False (Boolean).

    Returns:
        t_out: Time vector corresponding to the system's response (array-like).
        h_out: Scaled impulse response of the system (array-like).
    """
    t = np.linspace(0, T_total, int(Fs * T_total))
    duration = duration_ms / 1000
    onset_delay = onset_delay_ms / 1000
    omega_0 = 2 * np.pi * freq_hz
    a = 1 / duration

    num = [omega_0]
    den = [1, 2 * a, a ** 2 + omega_0 ** 2]
    
    system = TransferFunction(num, den)

    impulse = np.zeros_like(t)
    if not addnosie:
        impulse[int(Fs * onset_delay)] = 1

    N = int(Fs * duration)
    if addnosie:
        s = np.concatenate((np.random.randn(Ns), np.zeros(N - Ns)))
        impulse[int(Fs * onset_delay):int(Fs * onset_delay) + N] = s
    
    t_out, h_out, _ = lsim(system, U=impulse, T=t)

    h_out = amplitude * h_out

    return t_out, h_out

def segmentation(x, Fs, peak_find_boundary, wlen=800, verbose=False):
    """
    Segmentate the heart beat respones and indicate the S1 & S2 peaks in a plot.

    Parameters:
    x: Input of a 1D signal to segmentate (array-like).
    Fs: The sampling frequency of the signal (float).
    peak_find_boundary: The boundary of the amplitude of the threshold to find a peak (Threshold between:(a,b) or threshold under a) (float).
    wlen: Window lenth in samples for analyzing the promineces, default=800 (int).
    verbose: Printing the important values when True, default=False (Boolean).
    """
    t = np.linspace(0, len(x) /Fs, len(x)) # Define the time axis for the data that you want to segment

    peaks_index, _ = find_peaks(x, peak_find_boundary) 
    peaks_index_t = peaks_index / Fs # From samples index to time index
    prominences,leftb,rightb = peak_prominences(x, peaks_index, wlen=wlen)
    pp_height = x[peaks_index] - prominences # The height difference between the prominence and the peak
    width = np.diff(peaks_index) # This computes the difference between the peaks in amount of samples, note always len(width) = len(peaks_index) - 1

    S1 = []
    S2 = []
    for i in range(len(width)-1):
        if width[i] > width[i+1]:
            S2 = np.append(S2, peaks_index[i])
            S1 = np.append(S1, peaks_index[i-1])
        else:
            S2 = np.append(S2, peaks_index[i+1])
            S1 = np.append(S1, peaks_index[i])
        
    S1 = np.unique(S1).astype(int)
    S2 = np.unique(S2).astype(int)

    if verbose:
        print(f"Prominences value:{prominences}")
        print(f"Peak to prominence difference:{x[peaks_index]-prominences}")
        print(f"Samples of peaks measured:{len(peaks_index)}")
        print(f"Samples of width measured:{len(width)}")
        print(f"Peaks_index given (in samples):{peaks_index}")
        print(f"Peaks_index of S1:{S1}")
        print(f"Peaks_index of S2:{S2}")


    fig , ax = plt.subplots(4,1, figsize=(6, 10))
    ax[0].plot(t, x)
    ax[0].set_title("Signal response")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Amplitude")
    ax[0].plot(peaks_index_t, x[peaks_index], "x", label="Peaks")
    #ax[0].plot(peaks_index_t, prominences, "o", color="g")
    ax[0].plot(leftb / Fs, x[leftb], "o", color="b", label="Left-peak bases")
    ax[0].plot(rightb / Fs, x[rightb], "o", color="r", label="Right-peak bases")
    ax[0].scatter(S1/ Fs, x[S1],marker="d", c="y" ,label= "S1" )
    ax[0].scatter(S2/ Fs, x[S2],marker="d", c="k" ,label= "S2")
    ax[0].grid()
    ax[0].legend(fontsize=6)
    ax[1].stem(prominences)
    ax[1].set_title("Prominences of each peak")
    ax[1].set_xlabel("Sample index n")
    ax[1].set_ylabel("Amplitude")
    ax[2].stem(pp_height)
    ax[2].set_title("Peak to prominence height")
    ax[2].set_xlabel("Sample index n")
    ax[2].set_ylabel(" Amplitude height(difference)")
    ax[3].stem(width)
    ax[3].set_title("Width between peaks")
    ax[3].set_xlabel("Sample index n")
    ax[3].set_ylabel("Width in samples n")
    plt.tight_layout()
    plt.show()

def heartbeat_plot(M ,T, A, P, M_t, T_t, A_t, P_t, Fs, T_total, repetitions=1):
    """
    Plots the individual responses of valves and the repeated combined impulse response.

    Parameters:
    - M: Impulse response of valve M (array-like).
    - T: Impulse response of valve T (array-like).
    - A: Impulse response of valve A (array-like).
    - P: Impulse response of valve P (array-like).
    - Fs: Sampling frequency in Hz (float).
    - T_total: Total duration of one combined response in seconds (float).
    - repetitions: Number of times to repeat the combined response in the plot, default=1 (int).
    """
    if not (len(M) == len(T) == len(A) == len(P)):
        raise ValueError("The sizes of M, T, A, and P must be the same.") # Note that the size of all valves must be the same: 1D-array.
    combined_response = M + T + A + P
    combined_response_normalize = normalize(combined_response)
    repeated_response = np.tile(combined_response_normalize, repetitions)

    fig, ax = plt.subplots(4, 1, figsize=(6,10))
    ax[0].plot(M_t, M)
    ax[0].set_title("Valve M")
    ax[0].set_xlabel("Time [s]")
    ax[0].set_ylabel("Amplitude")
    ax[1].plot(T_t, T)
    ax[1].set_title("Valve T")
    ax[1].set_xlabel("Time [s]")
    ax[1].set_ylabel("Amplitude")
    ax[2].plot(A_t, A)
    ax[2].set_title("Valve A")
    ax[2].set_xlabel("Time [s]")
    ax[2].set_ylabel("Amplitude")
    ax[3].plot(P_t, P)
    ax[3].set_title("Valve P")
    ax[3].set_xlabel("Time [s]")
    ax[3].set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()

    plt.plot(np.linspace(0, T_total * repetitions, int(Fs * T_total * repetitions)), repeated_response)
    plt.title(f'Repeated Impulse Response of All Valves {repetitions} times')
    plt.xlabel('Time [s]')
    plt.ylabel('Amplitude (Normalized)')
    plt.show()

def propagation(x, G, delay, Fs):
    """
    Simulates the propagation of a signal with attenuation and delay.

    Parameters:
    - x: Input signal to be propagated (array-like or scalar).
    - G: Attenuation factor (float).
    - delay: Delay in seconds (float).
    - Fs: Sampling frequency in Hz (float).

    Returns:
    - x: Propagated signal with attenuation and delay applied (array-like).
    """
    if np.isscalar(x):
        x = np.array([x])
    x = np.pad(x,(int(delay*Fs),0))
    x = x * G
    return x

def attenuationANDdelay(z_v, z_m):
    """
    Calculates the attenuation and delay between two points based on their positions.

    Parameters:
    - z_v: Position of the source(valve) [x, y, z](array-like).
    - z_m: Position of the receiver(microphone) [x, y, z](array-like).

    Returns:
    - G: Attenuation factor (1 / distance) (float).
    - delay: Propagation delay in seconds (float).
    - d: Distance between the source and receiver (float).

    Notes:
    - The input z_v, z_m must in cm, as it is designed to calculate the output in small dimensions (human body).

    """
    z_v = z_v / 100 # Convert cm to m
    z_m = z_m /100  # Convert cm to m
    v = 60 # The velocity of sound in a body
    d = np.linalg.norm(z_v - z_m)
    G = 1 / d
    delay = (d / v)
    return G, delay, d

def multi_channel_signal(M, T, A, P, Fs, valves, microphones, verbose=False, plotting=False):
    """
    Simulates multi-channel signals from multiple valves to multiple microphones.

    Parameters:
    - M, T, A, P (array-like): Signals response from valves M, T, A, and P.
    - Fs (float): Sampling frequency in Hz.
    - valves (numpy.ndarray): 2D array representing positions of valves in cm.
    - microphones (numpy.ndarray): 2D array representing positions of microphones in cm.
    - verbose (bool, optional): If True, prints details for each valve-to-microphone propagation parameters.
    - plotting (bool, optional): If True, plots the resulting signals received for each microphone in one graph.

    Returns:
    - mic_signal (list of numpy.ndarray): Signals received at each microphone.
    - distance (numpy.ndarray): Distance matrix between valves and microphones.
    - G (numpy.ndarray): Attenuation matrix between valves and microphones.
    - delay (numpy.ndarray): Delay matrix between valves and microphones.
    """

    distance = np.empty((microphones.shape[0],valves.shape[0]))
    G = np.empty((microphones.shape[0],valves.shape[0]))
    delay = np.empty((microphones.shape[0],valves.shape[0]))
    mic_signal =[[] for _ in range(microphones.shape[0])]

    for i in range(microphones.shape[0]):
        for j in range(valves.shape[0]):

            G_value, delay_value, distance_value = attenuationANDdelay(valves[j], microphones[i])
            if verbose:
                print(f"valve[{j}] -- mic[{i}]:\n Attenuation:{G_value:.3f}, Delay:{(delay_value * 1000):.5f} ms, Distance:{(distance_value * 100):.2f} cm\n")
            G[i,j] = G_value
            delay[i, j] = delay_value
            distance[i, j] = distance_value

    for i, (G_r, delay_r) in enumerate(zip(G,delay)):
        M_pro = propagation(M, G_r[0], delay_r[0], Fs)
        T_pro = propagation(T, G_r[1], delay_r[1], Fs)
        A_pro = propagation(A, G_r[2], delay_r[2], Fs)
        P_pro = propagation(P, G_r[3], delay_r[3], Fs) 
        max_index = max(len(M_pro), len(T_pro), len(A_pro), len(P_pro))
        M_pro = np.pad(M_pro, (0,max_index - len(M_pro)))
        T_pro = np.pad(T_pro, (0,max_index - len(T_pro)))
        A_pro = np.pad(A_pro, (0,max_index - len(A_pro)))
        P_pro = np.pad(P_pro, (0,max_index - len(P_pro)))
        mic_signal[i] = M_pro + T_pro + A_pro + P_pro
        if verbose:
            print(f"The length at microphone channel {i} is {len(mic_signal[i])} samples")

    if plotting:
        plt.figure(figsize=(10, 6))  # Set figure size
        for i, row in enumerate(mic_signal):
            plt.plot(row, label=f"Signal {i+1}")  # Plot with a unique label for each row
        # Add legend, title, and grid
        plt.legend()
        plt.title("All Mic Signals")
        plt.xlabel("Sample Index")
        plt.ylabel("Amplitude")
        plt.grid()
        # Show the plot
        plt.show()

    return mic_signal, distance, G, delay 