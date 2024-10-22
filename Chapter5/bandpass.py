
import matplotlib.pyplot as plt 
import numpy as np 
import Chapter2.window
from matplotlib.lines import Line2D

C_2PI = 2 * np.pi
C_FFT_N = 32
C_FFT_FOLD_N =  np.floor_divide(C_FFT_N, 2) 
K_UNITY_SAMPLES = 7
K_OVER_2_UNITY_SAMPLES = np.floor_divide(K_UNITY_SAMPLES,2)

UNITY_GAIN_START_INDEX = -3
UNITY_GAIN_END_INDEX = 4
SAMPLES_4_PER_CYCLE = 4 

if __name__ == '__main__':
    fig = plt.figure() 
    fig.subplots_adjust( hspace=0.8 )
    ax1 = fig.add_subplot(6,1,1)
    ax2 = fig.add_subplot(6,1,2)
    ax3 = fig.add_subplot(6,1,3)
    ax4 = fig.add_subplot(6,1,4)
    ax5 = fig.add_subplot(6,1,5)
    ax6 = fig.add_subplot(6,1,6)

    # number unity values 
    m = np.arange(-C_FFT_FOLD_N + 1, C_FFT_FOLD_N  + 1)
    x_axis_sym_about_origin  = np.arange(-C_FFT_FOLD_N + 1, C_FFT_FOLD_N  + 1)
    x_axis = np.arange(0, C_FFT_N)
    x_axis_sym = np.arange(0, C_FFT_N-1)

    # filter frequency response  symmetric about origin  ( -fs/2 to fs/2 )
    H_m = np.zeros(shape=C_FFT_N)
    H_m[np.arange(UNITY_GAIN_START_INDEX + C_FFT_FOLD_N-1 , UNITY_GAIN_END_INDEX + C_FFT_FOLD_N-1 )] = 1
    H_m_fft = np.fft.fft(H_m)
    
    # filter frequency response over 0 to fs/2 
    H_m_0_to_fs = np.roll(H_m, -(np.abs(m[0]) - K_OVER_2_UNITY_SAMPLES)  - K_OVER_2_UNITY_SAMPLES   )
    m_0_to_fs = m + C_FFT_FOLD_N-1

    ax1.set_xticks (x_axis_sym_about_origin )
    ax1.scatter(x_axis_sym_about_origin , H_m, color='blue', s=4)
    ax1.vlines(x_axis_sym_about_origin , np.zeros(shape=m.size), H_m, color='blue', linewidth=1)
    ax1.set_title('H(m) -fs/2 to fs/2',  fontsize=6)
    ax1.tick_params(axis='x', labelsize=6)

    ax2.set_xticks ( x_axis )
    ax2.scatter(x_axis, H_m_0_to_fs, s=4, color='blue')
    ax2.vlines(x_axis , np.zeros(shape=m.size), H_m_0_to_fs, color='blue', linewidth=1)
    ax2.set_title('H(m) 0 to fs', fontsize=6)
    ax2.tick_params(axis='x', labelsize=6)
    
    hk_normal = np.fft.ifft(H_m_0_to_fs) 
    ax3.set_xticks ( x_axis )
    ax3.scatter(x_axis, hk_normal, s=4, color='blue')
    ax3.plot(x_axis, hk_normal, color='blue')
    ax3.tick_params(axis='x', labelsize=6)
    ax3.set_title('32 tap h(k) 0 to fs', fontsize=6)

    indices = np.hstack( (np.arange(C_FFT_FOLD_N+1, C_FFT_N) , np.arange(0, 1),  np.arange(1, C_FFT_FOLD_N)) )
    hk_normal_trunc =   hk_normal[indices]  
    hk_normal_trunc_fft = np.fft.fft(hk_normal_trunc).__abs__()
    ax4.plot(x_axis_sym, hk_normal_trunc , color='blue'  )
    ax4.scatter(x_axis_sym, hk_normal_trunc, s=4, color='blue')
    ax4.set_xticks ( x_axis_sym )
    ax4.tick_params(axis='x', labelsize=6)
    ax4.set_title('31 tap h(k) -fs/2 to fs/2', fontsize=6)

    x_trunc= np.arange(0 , C_FFT_N-1)
    shift_sinusoid = np.cos( (C_2PI  * x_trunc) * np.divide(C_FFT_N, SAMPLES_4_PER_CYCLE)  * np.divide(1, C_FFT_N) )
    hk_shifted = shift_sinusoid * hk_normal_trunc
    hk_shifted_fft = np.fft.fft(   hk_shifted   ).__abs__()
    indices = np.hstack( (np.arange(C_FFT_FOLD_N , C_FFT_FOLD_N + 1), np.arange(C_FFT_FOLD_N+1, C_FFT_N-1), np.arange(0, C_FFT_FOLD_N) ) )
    ax5.scatter(x_axis_sym, shift_sinusoid, s=4)
    ax5.tick_params(axis='x', labelsize=6)
    ax5.set_title('shift_signal - 4 samples per cycle', fontsize=6)
    ax5.tick_params(axis='x', labelsize=6)
    ax5.set_xticks ( x_axis_sym )

    x_axis_sym_zero_origin  =  x_axis_sym - (C_FFT_FOLD_N-1)
    ax6.set_xticks ( x_axis_sym - (C_FFT_FOLD_N-1) )

    ax6.tick_params(axis='x', labelsize=6)
    ax6.plot(x_axis_sym_zero_origin,hk_shifted_fft[indices] , color='orange', linewidth=1)
    ax6.plot(x_axis_sym_zero_origin,hk_normal_trunc_fft[indices], color='black', linewidth=1)
    legend_attr =dict(fontsize=5, loc=1)
    handles = [ Line2D([0], [0], label='31_tap bandpass freq_resp ', color='orange') , Line2D([0], [0], label='31_tap freq_resp', color='black') ] 
    ax6.legend(handles=handles, **legend_attr)

    plt.show()