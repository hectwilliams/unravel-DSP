import matplotlib.pyplot as plt 
import numpy as np 

C_2PI = np.pi * 2.0

def transfer_function_z(numerator_coef, denominator_ceof, number_points_per_axis = 1000):
    real_axis = np.linspace(-1, 1, number_points_per_axis)
    imag_axis =  np.linspace(-1, 1, number_points_per_axis)
    xx, yy = np.meshgrid(real_axis, imag_axis)
    z_ = xx + 1j*yy
    zz_numerator = np.zeros(shape=xx.shape, dtype=np.complex128)
    for i in range(numerator_coef.size-1, -1, -1):
        # use polynomial style 1)
        zz_numerator += numerator_coef[(numerator_coef.size-1) - i] * np.pow(z_, i, dtype=np.complex128)
    zz_denominator = np.zeros(shape=xx.shape, dtype=np.complex128)
    for i in range(denominator_ceof.size-1, -1, -1):
        zz_denominator += denominator_ceof[  (denominator_ceof.size-1) - i] * np.pow(z_, i, dtype=np.complex128)
    zz = np.divide(zz_numerator, zz_denominator)
    return xx, yy, zz

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2, projection= 'polar')
    ax3 = fig.add_subplot(3, 1 ,3, projection='3d')
    w = np.linspace(-np.pi, np.pi, 1000)
    rng = np.random.Generator(np.random.PCG64(42))

    numerator_coef = np.complex64 ( [0.9152, 0.1889 ] ) 
    denominator_coef = np.complex64 ( [1, 0.1127 ] ) 
    poles = np.roots(denominator_coef)
    zeros = np.roots(numerator_coef)
    zeros_resp = np.zeros(shape=w.shape,dtype=np.complex128 )
    poles_resp = np.zeros(shape=w.shape,dtype=np.complex128)
    for k,coef in enumerate(numerator_coef):
        zeros_resp += coef * np.exp(-1j * k *w)
    for k,coef in enumerate(denominator_coef):
        poles_resp += coef * np.exp(-1j * k *w)
    freq_resp = zeros_resp/poles_resp

    # IIR surface 
    xx, yy, zz = transfer_function_z(numerator_coef, denominator_coef, 1000)
    zz_abs = zz.__abs__() / zz.__abs__().max()
    filter_indices =  (zz_abs < 0.3)
    zz_abs_trim = filter_indices[filter_indices]

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
    ax3.plot(xx[filter_indices], yy[filter_indices], zz_abs[filter_indices],  color='gray')
    [ax3.tick_params(axis=ch, labelsize=3) for ch in list('xyz') ]
    ax3.set_xlabel('real' ,fontsize=6)
    ax3.set_ylabel('imag' ,fontsize=6)
    ax3.set_zlabel('|H(z)|' ,fontsize=6  )
    ax3.set_xlim(-1 , 1)
    ax3.set_ylim(-1, 1)

plt.show() 


