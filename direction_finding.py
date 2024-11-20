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

def a_lin(theta, M, d,v, f0):
