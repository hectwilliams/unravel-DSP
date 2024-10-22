import numpy as np 
import matplotlib.pyplot as plt 

C_2PI = 2*np.pi
C_FREQ = 1
C_ACQ_TIME = 1.1
C_SAMPLE_FREQ = 1000 # sample rate 
C_SAMPLE_PERIOD = pow(C_SAMPLE_FREQ, -1)  # sampel period ts
C_N = np.int32(C_ACQ_TIME / C_SAMPLE_PERIOD )
C_N_OVER_2 = np.int32(np.floor(C_N/2))
C_FOLD_FREQ = np.divide(C_SAMPLE_FREQ, 2.0)
C_FOLD_FREQ_FLOOR = np.int32(np.floor(C_FOLD_FREQ))

if __name__ == '__main__':
    amplitudes = np.hstack( (np.arange(0, 1, (1/400) * 2) , np.arange(1, 0,- (1/400) * 2)) )
    fig = plt.figure() 
    # time 
    n = np.arange(0, C_N)
    # generate some broadband signal
    y = np.zeros(shape=(C_N))
    for i, freq in  enumerate(np.linspace(1, 400)):
        a = amplitudes[i]
        y = y + a * np.cos(C_2PI *freq * n * C_SAMPLE_PERIOD )
    # frequency spectrum
    y_f = np.fft.fft(y)
    y_f_norm  =  np.divide(y_f, y_f.max()) # simple norm 
    y_f_norm_abs = np.abs(y_f_norm)
    y_f_db = 10* np.log10 ( y_f_norm_abs)
    f_n = (n/C_N) * C_SAMPLE_FREQ
    y_f_0_cemter = np.hstack( (y_f_norm_abs[C_N_OVER_2:], y_f_norm_abs[:C_N_OVER_2] ) ) 
    f_n_0_center=  (n - C_N_OVER_2) 
    # find spectrum max id 
    spectrum_grt_80_percent ,= np.nonzero( y_f_0_cemter > 0.8 )
    # upper half max 
    mid_point= np.int32( len(spectrum_grt_80_percent) * 0.5)
    id_max = spectrum_grt_80_percent[mid_point]
    # plots
    ax_1 = fig.add_subplot(2,2,1)
    ax_1.plot(n, y)
    ax_2 = fig.add_subplot(2,2,2)
    
    ax_2.plot(f_n, y_f_norm_abs, label='spectrum')
    ax_2.vlines([C_FOLD_FREQ_FLOOR], [0], [1], color='red', label='fold freq')
    ax_2.legend(loc="upper right")
    
    ax_3 = fig.add_subplot(2,1,2)
    ax_3.vlines(  f_n_0_center , np.zeros(shape= (C_N))    , y_f_0_cemter)
    
    plt.show()

