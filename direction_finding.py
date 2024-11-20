import matplotlib.pyplot as plt           # For plotting purposes
import numpy as np                        # For convolution function
import scipy
from scipy import signal                  # For filter function
from scipy.fft import fft, ifft           # For fft and ifft
import math                               # For numerical approximation of pi

#IP3 direction M3 M4 part 
#M is the number of microphones
# d is the distance between the microphones
#v is the speed of sound
# f0 is the center frequency of the wave in hertz

def a_lin(theta, M, d, v, f0):
    omega = f0/(math.pi)
    t = (d/v)*math.sin(theta)
    entries = np.linspace(0, M-1, M)
    
    vector = np.exp(-1j*omega*t*entries)
    s = np.exp(1j*omega*t)
    
    return vector

result = a_lin(0, 4, 0.1, 343, 1000)
print(result)
print("1")