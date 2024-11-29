import matplotlib.pyplot as plt
import numpy as np
import math
from scipy.io import wavfile
from scipy.fft import fft, ifft
from scipy.signal import convolve, unit_impulse, ShortTimeFFT

# #parameters
theta = 0 
M = 7      
v = 340  
f0 = 500  
delta = 0.5
wavelength = v / f0
d = wavelength * delta

def a_lin(theta, M, d, v, f0):
    omega = 2 * math.pi * f0  
    phase = (d / v) * np.sin(theta)
    entries = np.arange(M).reshape(M, 1)  
    vector = np.exp(-1j * omega * (phase * entries))
    return vector.flatten()

a_0 = a_lin(theta, M, d, v, f0)

theta0 = [0, 15]
SNR = 10
sigma_n = 10**(-SNR/20)
A = np.column_stack([a_lin(t, M, d, v, f0) for t in np.radians(theta0)])
A_H = A.conj().T
R = A @ A_H
Rn = np.eye(M,M)*sigma_n**2
Rx = R + Rn
print(A)

def matchedbeamformer(Rx, th_range, M, d, v, f0):
    spatial_response = []
    for theta in th_range:
        steering_vector = a_lin(theta, M, d, v, f0)
        response = steering_vector.conj().T @ Rx @ steering_vector
        normalized_response = response / M
        spatial_response.append(normalized_response)  
    
    return spatial_response
        

th_range = np.linspace(-np.pi/2, np.pi/2, 5000)
spatial_response = matchedbeamformer(Rx, th_range, M, d, v, f0)

# plt.figure(figsize=(10, 6))
# plt.plot(np.degrees(th_range), spatial_response)
# plt.title('Spatial Response for Matched Beamformer')
# plt.xlabel('Angle $\\theta$ (degrees)')
# plt.ylabel('Normalized $P_y(\\theta)$ (in terms of $M$)')
# plt.grid(True)
# plt.show()


def mvdr(Rx, th_range, M, d, v, f0):
   spatial_response2 = []
   inv_Rx = np.linalg.inv(Rx)  

   for theta in th_range:
        steering_vector = a_lin(theta, M, d, v, f0)
        numerator = 1
        denominator = steering_vector.conj().T @ inv_Rx @ steering_vector
        iloveee = numerator / denominator
        spatial_response2.append(iloveee)
   spatial_response2 = np.array(spatial_response2)
   return spatial_response2
   
th_range = np.linspace(-np.pi/2, np.pi/2,5000)
spatial_response2 = mvdr(Rx, th_range, M, d, v, f0)
#XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX
#[part for theta1]
theta_1 = theta0[1]
inv_Rx = np.linalg.inv(Rx)  
response1 = []


steering_vector1 = a_lin(theta_1 , M, d, v, f0)
numerator = 1
denominator2 = steering_vector1.conj().T @ inv_Rx @ steering_vector1
iloveee1 = numerator / denominator2
response1.append(iloveee1)

   
print(f'Response to second source (theta = {theta_1}°): {iloveee1:.6f}')

plt.figure(figsize=(10, 6))
plt.plot(np.degrees(th_range), spatial_response2)
plt.title('Spatial Response for MVDR')
plt.xlabel('Angle $\\theta$ (degrees)')
plt.ylabel('Normalized $P_y(\\theta)$ (in terms of $M$)')
plt.grid(True)
plt.show()
   

def p_y(theta, theta_0, M, d, v, f0):
    a_theta = a_lin(theta, M, d, v, f0)
    a_theta_reference = a_lin(theta_0, M, d, v, f0)
   
    a_theta_H = np.conjugate(a_theta).T
    
    p_y = (np.abs(np.matmul(a_theta_H, a_theta_reference))**2) 
    return p_y

theta_vals = np.linspace(-np.pi / 2, np.pi / 2, 1000)  
theta_0 = 0  

p_y_vals = [p_y(theta, theta_0, M, d, v, f0) for theta in theta_vals]

plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta_vals), p_y_vals)
plt.title('Spatial Response for Fixed Beamformer')
plt.xlabel('Angle $\\theta$ (degrees)')
plt.ylabel('Normalized $P_y(\\theta)$ (in terms of $M$)')
plt.grid(True)
plt.show()


# assignment 6.5.2



def narrowband_rX(data, fs, nperseg, noverlap):
    
    win = ('gaussian', 1e-2 * fs)  
    SFT = ShortTimeFFT.from_window(win, fs, nperseg, noverlap, scale_to='magnitude', phase_shift=None)
    f_bins = SFT.f
    #compute stft for all micrphone recordings
    Sx_all = []
    num_mics = data.shape[0]
    num_samples = data.shape[1]

    for i in range(data.shape[0]):
        Sx = SFT.stft(data[i, :])
        Sx_all.append(Sx)


    # Convert the list into a 3D NumPy array: (mics x frequency bins x time slices)
    Sx_all = np.array(Sx_all)    
    print(Sx_all.shape)  # Should print (num_mics, frequency_bins, time_slices)






    


    
   
    return Sx, f_bins


nperseg = 256
noverlap = 0
data = np.random.randn(4, 10000)
fs = 16000
aba = narrowband_rX(data, fs, nperseg, noverlap)

