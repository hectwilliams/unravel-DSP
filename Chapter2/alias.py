import numpy as np 
import matplotlib.pyplot as plt 
import time 

C_2PI = 2*np.pi 
C_SAMPLE_FREQ = 1000 # sample rate 
C_SAMPLE_PERIOD = pow(C_SAMPLE_FREQ, -1)  # sampel period ts
C_SIGNAL_FREQ_LOW = 300 
C_SIGNAL_FREQ_HIGH = 400
FFT_BIT_PER_SAMPLE = 12
FFT_N = np.pow(2, FFT_BIT_PER_SAMPLE)
C_FREQ_ANALYSIS = np.divide(C_SAMPLE_FREQ,  FFT_N)

def broadband_signal( sample_rate, n):
    ts= pow(sample_rate, -1)
    y = np.zeros(shape=(FFT_N))
    for i, f in  enumerate(np.linspace(C_SIGNAL_FREQ_LOW, C_SIGNAL_FREQ_HIGH, num=10000)):
        y = y + np.cos(C_2PI *f *  n * ts )
    return y
def update_fft(fs, dir):
    if dir == 1:
        fs += 10 
    if dir == -1:
        fs-= 10
    signal = broadband_signal(fs, n=n)
    signal_fft = np.fft.fft(signal)
    signal_fft_norm = np.divide(signal_fft, signal_fft.max())
    signal_fft_norm_abs = np.abs(signal_fft_norm)
    cond = (fs >= C_SIGNAL_FREQ_LOW*2  and fs >=C_SIGNAL_FREQ_HIGH*2 )
    color = 'black' if cond else 'red'
    alpha = 1 if cond else 0.4
    msg = f'fs={fs} fo={C_SIGNAL_FREQ_LOW}-{C_SIGNAL_FREQ_HIGH} \t' +  'good' if cond else 'aliasing, increase sample rate'
    p = ax_1.plot(n, signal_fft_norm_abs, alpha = alpha , color= color)
    ax_h, ax_w = ax_1.bbox.height, ax_1.bbox.width
    ax_1.annotate(  text=msg , xycoords='axes pixels', xy=(0, ax_h)  )
    plt.pause(0.01)
    ax_1.clear()
    return fs 

if __name__ == '__main__':
    fig = plt.figure() 
    ax_1 = fig.add_subplot(1,1,1)
    n = np.arange(0, FFT_N)
    while True:
        fs = 1000
        while fs > 100:
            fs = update_fft(fs,-1)
      
        while fs < 1000:
            fs = update_fft(fs,1)