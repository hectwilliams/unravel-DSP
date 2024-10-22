import numpy as np 
import matplotlib.pyplot as plt 
from matplotlib.lines import Line2D
from window import expander

C_SSB_N  = 32  # single side band N
C_L = C_SSB_N*2 # number of non padded values
C_N = C_L
C_2PI = np.pi * 2
C_SSB_N_DOUBLE = C_SSB_N * 2
C_SSB_N_TRIPLE = C_SSB_N * 3

def fft_pwr(data):
    # magnitude of fft 
    fft_data = np.fft.fft(data).__abs__()
    # bin power 
    data_fft = np.square(fft_data)
    return data_fft

if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(6, 1, 1)
    ax2 = fig.add_subplot(6, 1, 2)
    ax3 = fig.add_subplot(6, 1, 3)
    ax4 = fig.add_subplot(6, 1, 4)
    ax5 = fig.add_subplot(6, 1, 5)
    ax6 = fig.add_subplot(6, 1, 6)

    line_signal = Line2D([0], [0], label='signal', color='blue')
    line_noise = Line2D([0], [0], label='noise', color='skyblue')
    line_signal_noise = Line2D([0], [0], label='signal + noise', color='black')

    handles = [line_signal, line_noise, line_signal_noise]
    ax1.legend(handles=handles, fontsize=5, loc=1)
    ax2.legend(handles=handles, fontsize=5, loc=1)
    ax3.legend(handles=handles, fontsize=5, loc=1)
    ax4.legend(handles=handles, fontsize=5, loc=1)

    avg_padd_fft = np.zeros(shape=C_L)
    rng = np.random.Generator(np.random.PCG64(42))
    m = 6.5 
    y_signal_padd = np.array([])
    count = 0
    curr_range = np.arange(C_SSB_N, C_SSB_N*4)
    mem = np.zeros(shape=(curr_range.size, 2 * curr_range[-1]  ))
    mem2 = np.zeros(shape=(curr_range.size, 2 * curr_range[-1]  ))

    for ssb in curr_range:
        """ increasing the DFT size. Two cases are plotted: increases sample rate and fixed sample rate 
        """
        ax1.clear()
        ax2.clear()
        ax3.clear()
        ax4.clear()
        ax5.clear()

        ax1.legend(handles=handles, fontsize=5, loc=1)
        ax2.legend(handles=handles, fontsize=5, loc=1)
        ax3.legend(handles=handles, fontsize=5, loc=1)
        ax4.legend(handles=handles, fontsize=5, loc=1)

        ax1.set_ylabel('Amplitude' ,fontsize=5)
        ax1.set_xlabel('Bin Number', fontsize=5)
        ax2.set_xlabel( ax1.get_xlabel() ,  fontsize=5)
        ax2.set_ylabel('Bin Relative Power', fontsize=5)
        ax3.set_xlabel( ax1.get_xlabel() ,  fontsize=5)
        ax3.set_ylabel('Amplitude', fontsize=5)
        ax4.set_xlabel( ax1.get_xlabel() ,  fontsize=5)
        ax4.set_ylabel('Bin Relative Power dB',fontsize=5)
        ax4.set_ylim(-30 , 0)
        
        ax5.set_xlabel('Bin',fontsize=5)
        ax5.set_ylabel('Avg signal + noise',fontsize=5)
        ax6.set_xlabel('Bin ', fontsize=5)
        ax6.set_ylabel('Integration Avg FFT',fontsize=5)

        dsb = ssb*2 # double side band count
        n = np.arange(-ssb, ssb)
        a = np.sin(C_2PI * m * n * (1/(dsb)) )
        # signal 
        y = a  
        y_bin_pwr = fft_pwr(a)
        if ssb == C_SSB_N:
            y_signal_padd =  y 
        y_signal_padd = expander(y_signal_padd, dsb, mode=1)
        y_padd_bin_pwr = fft_pwr(y_signal_padd)
        
        # noise 
        y_noise = rng.standard_normal(size=dsb)
        # y_noise = rng.normal(0, 2, size=dsb)
        y_noise_bin_pwr = fft_pwr(y_noise)
        y_noise_padd = expander(y_noise[:C_SSB_N*2], dsb, mode=1)
        y_noise_padd_bin_pwr = fft_pwr(y_noise_padd)
        # noise + signal
        y_signal_noise = a + y_noise
        y_signal_noise_bin_pwr = fft_pwr(y_signal_noise)
        y_signal_noise_padd = y_signal_padd + y_noise_padd
        y_signal_noise_padd_bin_pwr = fft_pwr(y_signal_noise_padd)
        
        # find max (for relative plot)
        max_bin_pwr = np.max(np.hstack((y_bin_pwr, y_noise_bin_pwr, y_signal_noise_bin_pwr)))
        max_bin_pwr_padd = np.max(np.hstack((y_padd_bin_pwr, y_noise_padd_bin_pwr, y_signal_noise_padd_bin_pwr)))

    # unpadding data (sample rate will increase during simulation)
        y_bin_pwr_rel_ =  y_bin_pwr / max_bin_pwr
        y_noise_bin_pwr_rel =  y_noise_bin_pwr / max_bin_pwr
        y_signal_noise_bin_pwr_rel =  y_signal_noise_bin_pwr / max_bin_pwr
        
        # time signal 
        ax1.plot(n, y, linewidth=0.5, color='blue')
        ax1.plot(n, y_noise, linewidth=0.5, color='skyblue')
        ax1.plot(n , y_signal_noise, linewidth = 0.5, color='black')
        # spectrum signal 
        ax2.plot(y_bin_pwr_rel_[:ssb], color='blue', linewidth=0.5, alpha=0.3)
        ax2.plot(y_noise_bin_pwr_rel[:ssb], color='skyblue', linewidth=0.5)
        ax2.plot(y_signal_noise_bin_pwr_rel[:ssb], color='black', linewidth=0.5)
    # padding data (sample rate fixed; DFT increases vua zero padding 
        y_bin_pwr_padd_rel =  y_padd_bin_pwr / max_bin_pwr_padd
        y_bin_pwr_padd_rel_dB = 20 * np.log(y_bin_pwr_padd_rel)
        y_noise_padd_bin_pwr_rel = y_noise_padd_bin_pwr / max_bin_pwr_padd
        y_noise_padd_bin_pwr_rel_dB = 20 * np.log(y_noise_padd_bin_pwr_rel)
        y_signal_noise_padd_bin_pwr_rel =  y_signal_noise_padd_bin_pwr/ max_bin_pwr_padd
        y_signal_noise_padd_bin_pwr_rel_dB = 20 * np.log10(y_signal_noise_padd_bin_pwr_rel)
        # time signal ( zero padding)
        ax3.plot(y_signal_padd,  linewidth=0.5, color='blue' )
        ax3.plot(y_noise_padd,  linewidth=0.5, color='skyblue' )
        ax3.plot(y_signal_noise_padd,  linewidth=0.5, color='black' )
        # spectrum signal ( zero padding)
        ax4.plot(y_bin_pwr_padd_rel_dB[:ssb],  linewidth=0.5, color='blue' )
        ax4.plot(y_noise_padd_bin_pwr_rel_dB[:ssb],  linewidth=0.5, color='skyblue' )
        ax4.plot(y_signal_noise_padd_bin_pwr_rel_dB[:ssb],  linewidth=0.5, color='black' )

        fig.suptitle(f'DFT N = {dsb}')
        
        # avg_padd_fft = (y_signal_noise_padd + expander(avg_padd_fft, dsb, mode=1) ) 
        mem[count] = expander(y_signal_noise_padd, target_num_samples=   2 * curr_range[-1] , mode=1)  
        mem2[count] = expander(y_signal_noise_padd_bin_pwr_rel_dB, target_num_samples=   2 * curr_range[-1] , mode=1)  

        count += 1
        plt.pause(0.5)

    # print(avg_y_signal_padd)
    avg_y_signal_noise_padd = np.mean(mem, axis=0)
    ax5.plot(avg_y_signal_noise_padd[:C_N]  , linewidth=0.5, color='violet')
    
    avg_y_signal_noise_fft_db = np.mean(mem, axis=0)
    ax5.plot(avg_y_signal_noise_fft_db[:C_N]  , linewidth=0.5, color='violet')

    plt.show()





