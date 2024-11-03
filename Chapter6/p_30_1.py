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
pt0 = np.exp(1j * np.pi/120)
pt1 = np.exp(-1j * np.pi/120)

pole_roots = [ np.exp(1j * np.pi/120) , np.exp(-1j * np.pi/120)]
coef_poles_poly = np.poly(pole_roots)

zero_roots_table = [ 
    pt0,
    pt1, 
    pt0 * np.exp(1j * np.pi/4), 
    pt1 * np.exp(1j * np.pi/4), 
    pt0 * np.exp(1j *2* np.pi/4), 
    pt1 * np.exp(1j *2* np.pi/4), 
    pt0 * np.exp(1j *3* np.pi/4), 
    pt1 * np.exp(1j *3* np.pi/4), 
    pt0 * np.exp(1j *4* np.pi/4), 
    pt1 * np.exp(1j *4* np.pi/4), 

    pt0 * np.exp(-3j * np.pi/4), 
    pt1 * np.exp(-3j * np.pi/4), 
    
    pt0 * np.exp(-2j * np.pi/4), 
    pt1 * np.exp(-2j * np.pi/4), 

    pt0 * np.exp(-1j * np.pi/4), 
    pt1 * np.exp(-1j * np.pi/4), 
    ]

number = 0
step = 1
while True:
    
    number += step
    if number == 8:
        step = -1 
    if number == 1:
        step = 1 
    
    zero_roots = zero_roots_table[0: number*2]
    coef_zeros_poly = np.poly(zero_roots)

    poles_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k, coef in enumerate(coef_poles_poly):
        poles_resp += coef * np.exp(-1j * k * w)
    zeros_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k, coef in enumerate(coef_zeros_poly):
        zeros_resp += coef * np.exp(-1j * k * w)
    freq_response = zeros_resp / poles_resp
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
    for pole in np.roots(coef_poles_poly):
        ax2.plot( np.angle(pole), np.abs(pole),  marker='x', markersize=3 , color='black') 
    for zero in np.roots(coef_zeros_poly):
        ax2.plot( np.angle(zero), np.abs(zero),  marker='o', markersize=5, alpha=0.3, color='skyblue') 
    ax2.set_rmax( 2)
    plt.pause(0.5)