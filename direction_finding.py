import matplotlib.pyplot as plt
import numpy as np
import math

# IP3 direction M3 M4 part
# M: Number of microphones
# d: Distance between microphones
# v: Speed of sound
# f0: Center frequency of the wave in hertz
# (d/v) = delta
def a_lin(theta, M, d, v, f0):
    omega = 2*math.pi*f0
    phase = (d/v)*np.sin(theta)
    entries = np.arange(M)  
    vector = np.exp(-1j*omega*phase*entries)
    return vector

#create plots
theta = 0  
M = 7      
d = 2    
v = 340  
f0 = 500  


a_0 = a_lin(theta, M, d, v, f0)

def p_y(theta, theta_0, M, d, v, f0):
    # Calculate array response vectors
    a_theta = a_lin(theta, M, d, v, f0)
    a_theta_0 = a_lin(theta_0, M, d, v, f0)
    
    # Compute the Hermitian of a(theta): Conjugate transpose
    a_theta_H = np.conjugate(a_theta).T
    
    p_y = np.abs(np.dot(a_theta_H, a_theta_0))**2
    return p_y

# Define the range of theta values
theta_vals = np.linspace(-np.pi / 2, np.pi / 2, 100)  # Theta in radians
theta_0 = 0  # Reference angle in radians

# Compute P_y(theta) for all theta values
p_y_vals = [p_y(theta, theta_0, M, d, v, f0) for theta in theta_vals]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(np.degrees(theta_vals), p_y_vals)
plt.title('spatial response for fixed beamformer')
plt.xlabel('Angle $\\theta$ (degrees)')
plt.ylabel('$P_y(\\theta)$')
plt.grid(True)
plt.show()

