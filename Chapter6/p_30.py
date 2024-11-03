""" freq response from polynomial"""
import matplotlib.pyplot as plt 
import numpy as np 

C_2PI = np.pi * 2.0
MAG = 0.6
PHASE = np.pi/4

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, projection= 'polar')
    ax2.grid(True)
    ax2.set_yticklabels([])
    ax2.set_rmax( 1 )
    w = np.linspace(-np.pi, np.pi, 1000)
    rng = np.random.Generator(np.random.PCG64(42))
    """
        coef_bottom = [ 1, -2*np.cos(PHASE), 1]
        coef_top = []

        coef_bottom = [ 1, -(2*MAG)*np.cos(PHASE), np.square(MAG)]
        coef_top = [1, 1]
    """
    coef_top = [ 1 ,1 ]  
    coef_bottom = [ 1, -(2*MAG)*np.cos(PHASE), np.square(MAG) ]

    poles_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k, coef in enumerate(coef_bottom):
        poles_resp += coef * np.exp(-1j * k * w)
    zeros_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k, coef in enumerate(coef_top):
        zeros_resp += coef * np.exp(-1j * k * w)
    freq_resp = ( 1 if len(coef_top) == 0  else zeros_resp )  / (1 if len(coef_bottom) == 0  else poles_resp)
    freq_resp_abs = freq_resp.__abs__() 
    freq_resp_abs_norm = freq_resp_abs / freq_resp_abs.max()
    ax1.plot(w, freq_resp_abs_norm)
    for pole in np.roots(coef_bottom):
        ax2.plot( np.angle(pole), np.abs(pole),  marker='x', markersize=3) 
    for zero in np.roots(coef_top):
        ax2.plot( np.angle(zero), np.abs(zero),  marker='o', markersize=3, alpha=0.7, color='pink') 
    plt.show()