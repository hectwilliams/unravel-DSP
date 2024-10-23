
import matplotlib.pyplot as plt 
import numpy as np 

C_N = 13
C_N_OVER_2_UPPER = np.ceil(np.divide(C_N, 2))
C_N_OVER_2_FLOOR = np.floor_divide(C_N, 2)
C_ZEROS = np.zeros(shape=C_N)
C_TICKS = np.arange(0, C_N)
C_TICKS_MIRROR = C_TICKS - C_N_OVER_2_FLOOR

fig = plt.figure() 
ax1 = fig.add_subplot(5, 1, 1)
ax2 = fig.add_subplot(5, 1, 2)
ax3 = fig.add_subplot(5, 1, 3)
ax4 = fig.add_subplot(5, 1, 4)
ax5 = fig.add_subplot(5, 1, 5)

fig.subplots_adjust( hspace=0.8 )

passband = np.ones(2)
transband = np.ones(1) * 0.5
stopband = np.zeros(3)
center = np.ones(1)
ssb_no_zero = np.hstack(
    (
        passband, transband, stopband
    )
)
Hm_model = np.hstack( (ssb_no_zero[::-1] , center, ssb_no_zero) )
hk = np.fft.ifft(Hm_model)
ax1.scatter(C_TICKS_MIRROR, Hm_model, s=4)
ax1.vlines(C_TICKS_MIRROR,C_ZEROS,  Hm_model)
ax1.set_xticks ( C_TICKS_MIRROR)
ax1.tick_params(axis='x', labelsize=6)

ax2.scatter(C_TICKS_MIRROR, hk, s=4)
ax2.plot(C_TICKS_MIRROR, hk)
ax2.set_title( 'ifft H(m) --> h(k)' , fontsize=6)
ax2.set_xticks ( C_TICKS_MIRROR)
ax2.tick_params(axis='x', labelsize=6)

indices = np.hstack   ( (np.arange(C_N_OVER_2_FLOOR+1, C_N) , np.arange(0,1) , np.arange(1, C_N_OVER_2_FLOOR + 1)   ) ) 
ax3.scatter(C_TICKS_MIRROR, hk[indices], s=4)
ax3.plot(C_TICKS_MIRROR, hk[indices])
ax3.set_xticks ( C_TICKS_MIRROR)
ax3.tick_params(axis='x', labelsize=6)
ax3.set_title( 'ifft H(m) --> h(k)' , fontsize=6)

Hm_ = np.fft.fft(hk)
ax4.scatter(C_TICKS_MIRROR, Hm_.__abs__() , s=4)
ax4.vlines(C_TICKS_MIRROR, C_ZEROS , Hm_.__abs__() )
ax4.set_xticks ( C_TICKS_MIRROR)
ax4.tick_params(axis='x', labelsize=6)
ax4.set_title( 'fft h(k) --> H(m)'  , fontsize=6)


plt.show() 



