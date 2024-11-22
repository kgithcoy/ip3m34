import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, ifft
import math

# IP3 direction M3 M4 part
# M: Number of microphones
# d: Distance between microphones
# v: Speed of sound
# f0: Center frequency of the wave in hertz

def a_lin(theta, M, d, v, f0):
    omega = 2 * math.pi * f0
    phase_delay = (d / v) * np.sin(theta)
    entries = np.arange(M)  
    vector = np.exp(-1j * omega * phase_delay * entries)
    return vector

theta = 50  
M = 4      
d = 2    
v = 340  
f0 = 1000  

result = a_lin(theta, M, d, v, f0)
print(result)
