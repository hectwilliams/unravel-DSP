import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches

C_2PI = 2*np.pi
C_FREQ = 2
C_SAMPLE_FREQ = 1000 # sample rate 
C_FOLD_FREQ = np.int32(np.divide(C_SAMPLE_FREQ , 2)) # sample rate 
C_SAMPLE_PERIOD = np.divide(1.0,C_SAMPLE_FREQ)  # sampel period ts
C_FFT_BITS_PER_SAMPLE = 11
C_FFT_N = np.pow(2, C_FFT_BITS_PER_SAMPLE  )
C_FREQ_ANALYSIS = np.divide(C_SAMPLE_FREQ,  C_FFT_N)
C_FFT_N_OVER_2 = np.int32(C_FFT_N/2)
C_PLOT_SWITCH_TIME = 0.5
def cycles_per_FFT(signal_freq, m = 1):
    c = (signal_freq/C_SAMPLE_FREQ) * C_FFT_N
    c *= m
    return c
def center_fft(fft_y, n):
    y_f_0_cemter = np.hstack( (fft_y[C_FFT_N_OVER_2:], fft_y[:C_FFT_N_OVER_2] ) ) 
    f_n_0_center=  (n - C_FFT_N_OVER_2) *C_FREQ_ANALYSIS
    y_f_0_cemter =  y_f_0_cemter / y_f_0_cemter.max()
    return f_n_0_center, y_f_0_cemter
def approx_amplitude_response(curr_freq, m):
     cycles = cycles_per_FFT(curr_freq)
     u = np.pi * (cycles - m) * C_FREQ_ANALYSIS
     return (np.sin(u)/u) * 0.5 * C_FFT_N
def remove_plot(p):
    for line in p:
        line.remove() 
def new_signal(freq, id=0):
    cycles = cycles_per_FFT(freq)
    fig.suptitle(f'{np.round(freq, 2)}Hz    {np.round(cycles,2)} cycles ')
    y_leaky = np.cos(C_2PI*freq*n * C_SAMPLE_PERIOD)
    y_fft = np.fft.fft(y_leaky)
    freqs, data_leaky = center_fft(y_fft, n)
    zoom_range =  np.arange(C_FFT_N_OVER_2 - 50,C_FFT_N_OVER_2 + 50)
    sinc_response_pos = approx_amplitude_response(freq, n)
    freqs, data_sinc = center_fft(sinc_response_pos, n)
    attr = dict(color='black', alpha=0.5,linewidth=1) if id ==0 else dict(color='blue', alpha=1, linewidth = 1)
    p1 = ax1.plot(n_ts, y_leaky , **attr) 
    p2 = ax2.plot(freqs[zoom_range] , data_leaky[zoom_range] ,**attr)
    p3 = ax2.plot(freqs[zoom_range],  data_sinc[zoom_range],  **attr) 
    return p1, p2, p3
rng = np.random.Generator(np.random.PCG64(42))
fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
line1 = Line2D([0], [0], label='leaky input', color='blue')
line2 = Line2D([0], [0], label='non leaky input', color='black', alpha=0.5)
patch = mpatches.Patch(color='grey', label='some patch')
attr = dict(fontsize=5, loc=1)
ax1.legend(handles=[line1, line2,], **attr)
ax2.legend(handles=[line1, line2], **attr)
ax2.set_ylim(0, 1.1)
n = np.arange(0, C_FFT_N)
n_ts = n * C_SAMPLE_PERIOD
f_n = n * C_FREQ_ANALYSIS

for i in range(100):
    err = rng.uniform(low=C_FREQ_ANALYSIS, high=C_FREQ_ANALYSIS*2)
    freq_analysis  = C_FREQ_ANALYSIS*i
    plots_axis_0 = new_signal(freq_analysis, 0)
    plots_axis_1 = new_signal(freq_analysis + err, 1)
    plt.pause(C_PLOT_SWITCH_TIME)
    for p in plots_axis_0:
        remove_plot(p)
    for p in plots_axis_1:
        remove_plot(p)
plt.show()