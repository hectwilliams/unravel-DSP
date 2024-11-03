import matplotlib.pyplot as plt 
import numpy as np 

C_2PI = np.pi * 2.0

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2, projection= 'polar')
    w = np.linspace(0, C_2PI, 1000)
    rng = np.random.Generator(np.random.PCG64(42))

    numerator_coef = np.complex64 ( [2, 0 ] ) 
    denominator_coef = np.complex64 ( [1, np.exp(-0.88) ] ) 
    poles = np.roots(denominator_coef)
    zeros = np.roots(numerator_coef)
    zeros_resp = np.zeros(shape=w.shape,dtype=np.complex128 )
    poles_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k,coef in enumerate(numerator_coef):
        zeros_resp += coef * np.exp(-1j * k *w)
    for k,coef in enumerate(denominator_coef):
        poles_resp += coef * np.exp(-1j * k *w)
    freq_resp = zeros_resp/poles_resp
    freq_resp_mag = freq_resp.__abs__() 
    ax1.clear() 
    ax2.clear() 
    ax1.plot(w, freq_resp_mag) 
    ax2.grid(True)
    ax2.set_yticklabels([])
    ax2.set_rmax( 1 )
    for p in poles:
        ax2.plot( np.angle(p), np.abs(p),  marker='x', markersize=3) 
    for p in zeros:
        ax2.plot( np.angle(p), np.abs(p),  marker='o', markersize=3, alpha=0.3) 
    plt.show() 
