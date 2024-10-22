import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D

C_2PI = 2*np.pi
SSB_N  = 32  # single side band N
L = SSB_N*2 # number of non padded values

def rectangle( x_axis_space, start_index , end_index):
    data = np.zeros(shape=x_axis_space.size)
    for i in np.arange(start_index, end_index):
        idx = np.nonzero(x_axis_space == i)
        data[idx] = 1
    return data
def triangle(x_axis_space, start_index, end_index):
    window_size = end_index - start_index
    window_size_mid = np.floor((end_index + start_index)/2) + 0.0001
    data = np.zeros(shape=x_axis_space.size)
    indices = np.zeros(shape=window_size, dtype=np.int32)
    indices_i = 0
    for i in np.arange(start_index, end_index):
        x_idx, = np.nonzero(x_axis_space == i)
        if i >= start_index and i <= window_size_mid:
            data[x_idx] = i / window_size_mid
        elif i > window_size_mid and i < end_index:
            data[x_idx] = 2 - (i / window_size_mid)
        indices[indices_i] = int(x_idx)
        indices_i += 1
    # offset so base is equal to zero 
    min_value = np.abs(data[indices].min())
    data[indices] += min_value
    return data/ data.max()
def hanning(x_axis_space, start_index, end_index, is_hamming= False):

    data = np.zeros(shape=x_axis_space.size)
    window_size = end_index - start_index
    indices = np.zeros(shape=window_size, dtype=np.int32)
    indices_i = 0
    for n in np.arange(start_index, end_index):
        idx , = np.nonzero(x_axis_space == n)
        if not is_hamming:
            data[idx] = 0.5 * (1 - np.cos(C_2PI* n *( 1/window_size) ))
        if  is_hamming:
            data[idx] = 0.54 - 0.46 * np.cos(C_2PI*n*(1/window_size))
        indices[indices_i] = int(idx)
        indices_i += 1
    min_value = np.abs(data[indices].max() - data[indices].min())
    data[indices] *= -1
    data[indices]  += min_value 
    return data/ data.max()
def hamming (x_axis_space, start_index,  end_index):
    return hanning(x_axis_space, start_index, end_index, True)
def normalize(data: np.ndarray):
    return data / data.max()
def expander(data: np.ndarray, target_num_samples, mode = 0):
    """Zero pads data alternating 1 zero on either side until target_num_samples is reacged
        mode = 0 - both sides 
        mode - 1 - right side 
        mode - 2 - left side 
    """
    if target_num_samples < data.size :
        raise ValueError('target size must be >= to data size')
    data_new = data.copy()
    sel = 0
    while target_num_samples > data_new.size:
        if not sel or mode  == 1:
            data_new = np.hstack( (data_new, np.zeros(shape=1)) )
        elif set or self  == 2:
            data_new = np.hstack( ( np.zeros(shape=1) , data_new ) )
        sel = sel ^ 1
    return data_new

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(3,1,1)
    ax2 = fig.add_subplot(3,1,2)
    ax3 = fig.add_subplot(3,1,3)


    x = (-SSB_N, SSB_N) # dft size 
    m = 9.0  # cycle per dft 
    x_active_range = np.arange(-SSB_N, SSB_N)

    w0_init = rectangle(x_active_range, *x)
    w1_init = triangle(x_active_range, *x)
    w2_init = hanning(x_active_range, *x)
    w3_init = hamming(x_active_range, *x)
    x_signal_init = np.sin(C_2PI*m * x_active_range * (1/(SSB_N*2))) 

    # pass windows and signal thru expander (zero pad if neccesary)
    for a in np.arange(SSB_N*2, 129 ):
        
        ax1.clear()
        ax2.clear()
        ax3.clear()

        N = a
        w0 = expander(w0_init, N, mode=1)
        w1 = expander(w1_init, N, mode=1)
        w2 = expander(w2_init, N, mode=1)
        w3 = expander(w3_init, N, mode=1)
        x_signal = expander(x_signal_init, N, mode=1)
        attrs = [
            dict(color='blue'),
            dict(color='skyblue'),
            dict(color='gray', alpha=0.6),
            dict(color='black', linewidth=0.3, linestyle='dashdot' )
        ]
        line_rect = Line2D([0], [0], label='rectangle', **attrs[0])
        line_tri = Line2D([0], [0], label='triangle', **attrs[1])
        line_han = Line2D([0], [0], label='hanning', **attrs[2])
        line_ham = Line2D([0], [0], label='hamming', **attrs[3])
        line_signal = Line2D([0], [0], label=f'{m} cycle signal', color='pink', linewidth=0.3)

        handles = [line_rect, line_tri, line_han, line_ham, line_signal]
        
        ffts = [np.zeros(shape=(SSB_N*2))  for _ in range(4)]
        largest_fft_sample = np.finfo(np.float64).min
        windows = [w0, w1, w2, w3]
        
   
        # find max magnitude of all windows combined
        for i, w in enumerate(windows):
            ffts[i] = np.abs(np.fft.fft(w ))
            if ffts[i].max() > largest_fft_sample:
                largest_fft_sample = ffts[i].max()
    
         # some input signal 
        x_signal_fft = np.abs(np.fft.fft(x_signal))
        x_signal_fft_norm = (x_signal_fft / largest_fft_sample) +  0.00001
        x_signal_fft_dB = 20 * np.log10(x_signal_fft_norm)
        
        fft_main_lobe_index = np.argmax(x_signal_fft_norm[:np.int32(np.divide(N ,2))])
        print( f'main_lobe index {fft_main_lobe_index} \t main_lobe_value  {x_signal_fft[fft_main_lobe_index]}')

        fig.suptitle(f' TOTAL_NUM_SAMPLES = {a}   DFT_NUM_SAMPLES = {L} \n m/N = {m}/{L,}={m/L}  \n New (m/N) from padding = {fft_main_lobe_index}/{N} = {(fft_main_lobe_index/N).round(5) }' , fontsize=8)

        ax1.plot(x_signal, linewidth = 0.3, color='pink')
        ax2.plot(x_signal_fft[:32], linewidth = 0.3, color='pink')
        ax3.plot(x_signal_fft_dB[:32], linewidth = 0.3, color='pink')

        # find max magnitude of all windows combined
        for i, w_fft in enumerate(ffts):
            w_fft_norn = ( w_fft / largest_fft_sample) + 0.0099
            w_fft_dB = 20 * np.log10(w_fft_norn)
            legend_attr =dict(fontsize=5, loc=1)
            # sample time plot
            ax1.plot(windows[i], **attrs[i])
            ax1.legend(handles=handles, **legend_attr)
            # spectrum plot
            ax2.plot(w_fft[:32], **attrs[i])
            ax2.legend(handles=handles, **legend_attr)
            ax2.set_xticks (np.arange(0, 32)  )
            ax2.tick_params(axis='x', labelsize=7)
            # spectrum relative magnitudes in dB
            ax3.plot(w_fft_dB[:32], **attrs[i])
            ax3.legend(handles=handles, **legend_attr)
            ax3.set_xticks (np.arange(0, 32)  )
            ax3.tick_params(axis='x', labelsize=7)
            ax3.set_ylim(-40, 0)

    # signal 
        plt.pause(0.1)
    plt.show()


