import matplotlib.pyplot as plt 
import numpy as np 

C_2PI = np.pi * 2.0

fig = plt.figure()
fig.subplots_adjust( hspace=0.4 )
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2, projection= 'polar')
ax2.grid(True)
ax2.set_yticklabels([])
ax2.set_rmax( 2)
w = np.linspace(-np.pi, np.pi, 1000)
pt = np.exp(1j * np.pi/8)
pole_roots = []
coef_poles_poly = np.poly(pole_roots)
zero_unit_circle_table = np.array([pt for _ in range(15)])
zero_not_unit_circle_table = np.array([pt for _ in range(32)])
# unit circle (excluding z = 1)
pt = np.exp(1j * np.pi/8)
for i in range(0, 15):
    zero_unit_circle_table[i] = np.pow(pt, i + 1)
# non unit ciecle zeros
pt = np.exp(1j * np.pi/16)
step = 0.1
delta  = 0.0
while True:
    delta = delta + step
    if delta >= 0.5:
        step = -0.1
        delta = 0.5
    elif delta <= 0.0:
        step = 0.1
        delta = 0.0
    for i in np.arange(start=0, stop=32, step=2):
        amp_higher = 1 + delta
        amp_lower = 1 - delta
        zero_not_unit_circle_table[i ] = amp_lower * np.pow(  pt, i )
        zero_not_unit_circle_table[i + 1] = amp_higher * np.pow( pt, i)
    # zero_roots = np.hstack( (zero_not_unit_circle_table ,zero_unit_circle_table ) )
    zero_roots = zero_not_unit_circle_table
    coef_zeros_poly = np.poly(zero_roots)
    zeros_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k, coef in enumerate(coef_zeros_poly):
        zeros_resp += coef * np.exp(-1j * k * w)
    freq_response = zeros_resp 
    freq_response_abs = freq_response.__abs__()
    freq_response_norm = freq_response_abs / freq_response_abs.max() 
    ax1.clear()
    ax2.clear() 
    ax2.grid(True)
    ax2.set_yticklabels([])
    ax1.set_title('Frequency Response'  ,  fontsize=10)
    ax1.set_xlabel('Radian/Sample',  fontsize=10)
    ax1.set_ylabel('Magnitude',  fontsize=10)
    ax1.set_ylim(0, 1.2)
    ax1.fill_between(w, freq_response_norm, edgecolor='black', linewidth=4, facecolor='skyblue')
    for zero in np.roots(coef_zeros_poly):
        ax2.plot( np.angle(zero), np.abs(zero),  marker='o', markersize=3, alpha=0.3, color='skyblue') 
    ax2.set_rmax( 3)
    plt.pause(0.5)
plt.show()