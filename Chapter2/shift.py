import numpy as np 
import matplotlib.pyplot as plt 

C_2PI = 2*np.pi
C_FREQ = 2
C_SAMPLE_FREQ = 1000 # sample rate 
C_FOLD_FREQ = np.int32(np.divide(C_SAMPLE_FREQ , 2)) # sample rate 
C_SAMPLE_PERIOD = np.divide(1.0,C_SAMPLE_FREQ)  # sampel period ts
C_FFT_10_BIT_PER_SAMPLE = 10
C_FFT_N_1024 = np.pow(2, C_FFT_10_BIT_PER_SAMPLE  )
C_FREQ_ANALYSIS_1024 = np.divide(C_SAMPLE_FREQ,  C_FFT_N_1024)

def zoom_fft(data, sweep_distance = 20, index = 0):
    """collects samples closest and equal distance from sample at location index (set by sweep distance)"""
    if index - sweep_distance < 0:
        return np.arange(0, sweep_distance)
    if index + sweep_distance > data.size:
        return np.arange(data.size - sweep_distance, data.size)
    mid = np.int32(np.floor(np.divide(sweep_distance, 2.0)))
    return np.arange(index - mid, index + mid + 1)

def get_fft(f, n, n_freq, component_id):
    k = 3
    y = np.cos(C_2PI * f  * n * C_SAMPLE_PERIOD )
    y_fft = np.fft.fft(y)
    y_fft_norm = np.divide(y_fft, y_fft.max())
    y_fft_phase = np.angle(y_fft) 
    indices =  zoom_fft(y_fft_norm , sweep_distance = 50, index = component_id)
    
    n_shift =  n + k 
    y_shift = np.cos(C_2PI * f  * n_shift * C_SAMPLE_PERIOD )
    y_shift_fft = np.fft.fft(y_shift)
    y_shift_fft_norm = np.divide(y_shift_fft, y_shift_fft.max())
    y_shift_fft_phase = np.angle(y_shift_fft) 

    # signal 
    ax1.plot(n *C_SAMPLE_PERIOD, y, color = 'skyblue')
    w, h = ax1.bbox.width, ax1.bbox.height
    ax1.annotate( text='sinusoid', textcoords='axes pixels', xy=(0.5, 1) ,fontsize=6 )
    # signal shift
    ax2.plot(n_shift *C_SAMPLE_PERIOD, y_shift, color = 'yellow')
    # magnitude (should be equal)
    ax3.plot(n_freq ,np.abs(y_fft_norm), color='skyblue', alpha=0.8) 
    ax3.plot(n_freq ,np.abs(y_shift_fft_norm), color='yellow', alpha=1, linestyle='-') 
    # phase plot 
    ax4.plot(n_freq, y_fft_phase, linewidth=0.5, color='skyblue') 
    ax4.plot(n_freq, y_shift_fft_phase, linewidth=0.5, color='yellow')   # FFT of real signal shifted ahead in time 
    # phase diff  
    ax5.plot(n_freq, y_fft_phase - y_shift_fft_phase, linewidth=0.5, color='violet')  # phase difference beteen signal and shifted_signal; notice the phase relationship X_phase(m) = - X_phase(N - m)

    fig.suptitle(f' tone-id {f.round(3)} ')
    plt.pause(0.5)
    ax5.clear()
    ax3.clear()
    ax2.clear()
    ax1.clear()
    ax4.clear()

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(5, 1, 1)
    ax2 = fig.add_subplot(5, 1, 2)
    ax3 = fig.add_subplot(5, 1, 3)
    ax4 = fig.add_subplot(5, 1, 4)
    ax5 = fig.add_subplot(5, 1, 5)
    n_1024  = np.arange(0, C_FFT_N_1024) 
    f_n = n_1024 * C_FREQ_ANALYSIS_1024 
    for i in range(C_FFT_N_1024):
        f = C_FREQ_ANALYSIS_1024 * i
        get_fft(f, n_1024, f_n, i)
    plt.show()
