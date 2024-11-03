import matplotlib.pyplot as plt 
import numpy as np 

C_2PI = np.pi * 2.0

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2, projection= 'polar')
w = np.linspace(-np.pi, np.pi, 1000)
rng = np.random.Generator(np.random.PCG64(42))

for _ in range(100):
    K = 1 # rng.standard_normal()
    numerator_coef = np.complex64 ( [-K, 1] ) 
    denominator_coef = np.complex64 ( [1, -K ] ) 

    poles = np.roots(denominator_coef)
    zeros = np.roots(numerator_coef)

    zeros_resp = np.zeros(shape=w.shape,dtype=np.complex128 )
    poles_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k,coef in enumerate(numerator_coef):
        zeros_resp += coef * np.exp(-1j * k *w)
    for k,coef in enumerate(denominator_coef):
        poles_resp += coef * np.exp(-1j * k *w)
    freq_resp = zeros_resp/poles_resp
    ax1.clear() 
    ax2.clear() 
    ax1.plot(w, 20 * np.log(freq_resp.__abs__()/freq_resp.__abs__().max()) )
    ax2.grid(True)
    ax2.set_yticklabels([])
    ax2.set_rmax( 1 )
    for p in poles:
        ax2.plot( np.angle(p), np.abs(p),  marker='x', markersize=3) 
    for p in zeros:
        ax2.plot( np.angle(p), np.abs(p),  marker='o', markersize=3, alpha=0.3) 
    plt.pause(0.5)
plt.show() 


