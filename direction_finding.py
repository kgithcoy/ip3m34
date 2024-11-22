import matplotlib.pyplot as plt
import numpy as np
import math

#parameters
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
    return vector


a_0 = a_lin(theta, M, d, v, f0)
print(a_0)


theta0 = np.array([0, 15])
SNR = 10
sigma_n = 10**(-SNR/20)
A = a_lin(theta0, M, d, v, f0)
A_H = A.conj().T
R = A @ A_H
Rn = np.eye(M,M)*sigma_n**2
Rx = R + Rn


def matchedbeamformer(Rx, th_range, M,d,v,f0):
    
    return 1

def mvdr(Rx, th_range, M,d,v,f0):
    return 1



# def p_y(theta, theta_0, M, d, v, f0):
#     a_theta = a_lin(theta, M, d, v, f0)
#     a_theta_reference = a_lin(theta_0, M, d, v, f0)
   
#     a_theta_H = np.conjugate(a_theta).T
    
#     p_y = (np.abs(np.matmul(a_theta_H, a_theta_reference))**2) 
#     return p_y

# theta_vals = np.linspace(-np.pi / 2, np.pi / 2, 100)  
# theta_0 = 0  

# p_y_vals = [p_y(theta, theta_0, M, d, v, f0) for theta in theta_vals]

# plt.figure(figsize=(10, 6))
# plt.plot(np.degrees(theta_vals), p_y_vals)
# plt.title('Spatial Response for Fixed Beamformer')
# plt.xlabel('Angle $\\theta$ (degrees)')
# plt.ylabel('Normalized $P_y(\\theta)$ (in terms of $M$)')
# plt.grid(True)
# plt.show()
