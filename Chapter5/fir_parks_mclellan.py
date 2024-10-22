
"""

    Freq Reponse Definition 

    # # # # # #
               #
                #
                 #
                  #
                   # # # # # # # # # # #           

"""
import matplotlib.pyplot as plt 
import numpy as np 
import Chapter2.window
from matplotlib.lines import Line2D

C_SSB = 20
C_DSB_SIZE = C_SSB*2  + 1
C_ZEROS = np.zeros(shape=C_DSB_SIZE)
C_TICKS = np.arange(0, C_DSB_SIZE)
if __name__ == '__main__':
    #figure 
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)

    # desired freq response 
    passband = np.ones(shape=5)
    transband = np.arange(1, 0, -(1/5))[1:]
    stopband = np.zeros(shape=11)
    ssb = np.hstack( (passband, transband, stopband))
    dsb =  np.hstack  ( ( ssb[::-1]   , np.ones(shape=1) , ssb ))
    x = np.arange(0, C_DSB_SIZE)
    x = x - (C_SSB)

    ax1.vlines(x, C_ZEROS, dsb)
    ax1.scatter(x, dsb, s= 3)
    ax1.set_title('Define H(m) -fs/2 to fs/2',  fontsize=6)
    ax1.tick_params(axis='x', labelsize=6)
    ax1.set_xticks (  x ) 

    # inverse fft 
    hk = np.fft.ifft(dsb)
    indices = np.hstack   ( (np.arange(C_SSB+1, C_DSB_SIZE) , np.arange(0,1) , np.arange(1, C_SSB + 1)   ) ) 
    ax2.set_title('ifft( H(m) ) --> h(k)',  fontsize=6)
    ax2.tick_params(axis='x', labelsize=6)
    ax2.set_xticks ( x)
    ax2.scatter(x, hk[indices], color = 'skyblue', s=4)
    ax2.vlines(x, C_ZEROS, hk[indices])
    # forwarf fft 
    Hm_ = np.fft.fft(hk).__abs__()
    ax3.scatter(x, Hm_, s= 3)
    ax3.vlines(x, C_ZEROS, Hm_)
    ax3.set_title( 'fft( h(k) --> H(m)' , fontsize=6)
    ax3.set_xticks ( x)
    ax3.tick_params(axis='x', labelsize=6)

    
    plt.show() 