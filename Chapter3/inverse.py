"""invert dft using forward dft routine"""
import numpy as np 
import matplotlib.pyplot as plt 

C_2PI = 2*np.pi
C_SSB_N  = 16  # single side band N
C_DSB_N = C_SSB_N*2 # number of non padded values

if __name__ == '__main__':
    fig = plt.figure()
    fig.tight_layout()
    fig.subplots_adjust( hspace=0.4 )
    ax1 = fig.add_subplot(4, 1 ,1)
    ax2 = fig.add_subplot(4, 1 ,2)
    ax3 = fig.add_subplot(4, 1 ,3)
    ax4 = fig.add_subplot(4, 1 ,4)
    # time axis 
    n = np.arange(-C_SSB_N, C_SSB_N)
    # broadband signal 
    y = np.zeros(shape=C_DSB_N) + 0.2
    y_avg = y.mean()
    y = y - y_avg
    # fft 
    f_n = np.arange(0, C_DSB_N)
    for n_i in np.arange(2,8):
        y = y + np.cos(C_2PI * n_i * np.divide(1, C_DSB_N) * n )
    ax1.plot(n, y)
    ax1.set_title('time sample data', fontsize=6)
    # fft
    y_fft = np.fft.fft(y)
    y_fft_bin_pwr = np.square(y_fft)
    ax2.plot(f_n, y_fft_bin_pwr)
    ax2.fill_between(f_n, y_fft_bin_pwr, 0, alpha=0.3)
    print('axes',ax2.bbox.width, ax2.bbox.height)
    print('fig', fig.bbox.width, fig.bbox.height)
    ax2.set_title('spectrum', fontsize=6)
    #inverse 
    y_real_compliment = np.fft.fft(y_fft.real)
    y_imag_compliment = np.fft.fft(-1*y_fft.imag)
    y_imag= ( y_imag_compliment * -1 ) / C_DSB_N
    y_real = y_real_compliment / C_DSB_N
    y_recover = y_real + 1j*y_imag
    ax3.plot(n, y_recover.real )
    ax3.set_title('inverse via forward dft', fontsize=6)
    ax4.plot  (n, np.abs(y_recover.real - y ) )
    ax4.set_title('time sample data residuals', fontsize=6)
    ax4.set_ylim(-0.5, 0.5)
    plt.show()