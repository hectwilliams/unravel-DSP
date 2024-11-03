import matplotlib.pyplot as plt 
import numpy as np 
C_2PI = np.pi * 2.0

w = np.linspace(-np.pi, np.pi, 1000)

numerator = 1 + 2*np.exp(-1j * w)
denominator = 1 - 2*np.exp(-3j * w)

numerator_coef = np.complex64 ( [1, 0] )  # Z + 0
w_resonator_deg = 180/2
denominator_coef = np.complex64 ( [1, np.exp(1j * np.deg2rad(w_resonator_deg)) ] ) # Z + 1
poles = np.roots(denominator_coef)
zeros = np.roots(numerator_coef)

zeros_resp = np.zeros(shape=w.shape,dtype=np.complex128 )
poles_resp = np.zeros(shape=w.shape,dtype=np.complex128)
for k,coef in enumerate(numerator_coef):
    zeros_resp += coef * np.exp(-1j * k *w)
for k,coef in enumerate(denominator_coef):
    poles_resp += coef * np.exp(-1j * k *w)
freq_resp = zeros_resp/poles_resp

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2, projection= 'polar')
ax1.plot(w, freq_resp.__abs__())
ax2.grid(True)
ax2.set_yticklabels([])
ax2.set_rmax( 1 )
for p in poles:
    ax2.plot( np.angle(p), np.abs(p),  marker='x', markersize=3) 
for p in zeros:
    ax2.plot( np.angle(p), np.abs(p),  marker='o', markersize=3) 

plt.show() 


