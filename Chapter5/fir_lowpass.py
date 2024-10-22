
import matplotlib.pyplot as plt 
import numpy as np 
import Chapter2
import Chapter2.window
"""
* Register python path prior to export
    export PYTHONPATH="/Path-to-root-directory"
"""

C_2PI = 2 * np.pi
C_FFT_N = 32
C_FFT_FOLD_N =  np.floor_divide(C_FFT_N, 2) 

def analysis_window( size, indices, axes):
    """ plot pievcewise window 
    
    Args:
        size - size of symmetric x axis
        indices - indices ( or x-axis points with unity gain)
        axes -  axis to plot window
    
    Returns:
        Line2D object
    """
    size_half = np.floor_divide(size , 2)
    x = np.arange( -size_half, size_half  + 1)
    base_indices = np.zeros(shape=2)
    b_walk = 0
    for i,  x_index in enumerate(x):
        if x_index in indices:
            if b_walk == 0:
                base_indices[b_walk] = i
                b_walk += 1
            elif b_walk == 1:
                base_indices[b_walk] = i
    base_indices = np.hstack((base_indices, base_indices))
    return axes.fill_between( base_indices[ : 2 ] , 0 , 1, edgecolor='black' , color='skyblue', alpha=0.4)
def blackman_window(indices, size):
    y = 0.42 - (0.5 * np.cos(C_2PI*indices/(size -1))) + (0.08* np.cos(C_2PI*2*indices/(size -1) )    )  
    y = Chapter2.window.expander(y, size)
    return y 

if __name__ == '__main__':
    fig = plt.figure() 
    ax1 = fig.add_subplot(7,1,1)
    ax2 = fig.add_subplot(7,1,2)
    ax3 = fig.add_subplot(7,1,3)
    ax4 = fig.add_subplot(7,1,4)
    ax5 = fig.add_subplot(7,1,5)
    ax6 = fig.add_subplot(7,1,6)
    ax7 = fig.add_subplot(7,1,7)

    fig.subplots_adjust( hspace=1.4 )

    #number unity values 
    K = 7
    K_over_2 = np.floor_divide(K,2)
    unity_gain_start_index = -3
    unity_gain_emd_index = 4
    m = np.arange(-C_FFT_FOLD_N + 1, C_FFT_FOLD_N  + 1)
    x_axis_sym_about_origin  = np.arange(-C_FFT_FOLD_N + 1, C_FFT_FOLD_N  + 1)
    x_axis = np.arange(0, C_FFT_N)
    x_axis_sym = np.arange(0, C_FFT_N-1)

    # filter frequency response  symmetric about origin  ( -fs/2 to fs/2 )
    H_m = np.zeros(shape=C_FFT_N)
    H_m[np.arange(unity_gain_start_index + C_FFT_FOLD_N-1 , unity_gain_emd_index + C_FFT_FOLD_N-1 )] = 1
    H_m_fft = np.fft.fft(H_m)

    # filter frequency response over 0 to fs/2 
    H_m_0_to_fs = np.roll(H_m, -(np.abs(m[0]) - K_over_2)  - K_over_2   )
    m_0_to_fs = m + C_FFT_FOLD_N-1
    ax1.set_xticks (x_axis_sym_about_origin )
    ax1.scatter(x_axis_sym_about_origin , H_m, color='blue', s=4)
    ax1.vlines(x_axis_sym_about_origin , np.zeros(shape=m.size), H_m, color='blue', linewidth=1)
    ax1.set_title('H(m) -fs/2 to fs/2',  fontsize=6)
    ax1.tick_params(axis='x', labelsize=6)

    ax2.set_xticks ( x_axis )
    ax2.scatter(x_axis, H_m_0_to_fs, s=4)
    ax2.vlines(x_axis , np.zeros(shape=m.size), H_m_0_to_fs, color='blue', linewidth=1)
    ax2.set_title('H(m) 0 to fs', fontsize=6)
    ax2.tick_params(axis='x', labelsize=6)

    hk_normal = np.fft.ifft(H_m_0_to_fs  ) 
    ax3.plot(hk_normal)
    ax3.set_title(f'h(k) {C_FFT_N} tap FIR Filter' , fontsize=6)
    ax3.set_xticks ( x_axis )
    ax3.tick_params(axis='x', labelsize=6)
    ax3.scatter(x_axis, hk_normal, s=4)

    # elimate middle sample to provude symmetric fir filter 
    hk_normal_sym = np.hstack ( (hk_normal[:C_FFT_FOLD_N] , hk_normal[C_FFT_FOLD_N+1:] ) )
    H_m_0_to_fs_inverse_fft_shift = np.roll( hk_normal_sym , C_FFT_FOLD_N - 1)
    ax4.set_title(f'h(k) {C_FFT_N - 1} tap FIR Filter (symmetric)' , fontsize=6)
    ax4.plot(H_m_0_to_fs_inverse_fft_shift)
    ax4.set_xticks ( x_axis_sym )
    ax4.scatter(x_axis_sym , H_m_0_to_fs_inverse_fft_shift, s=4)
    ax4.tick_params(axis='x', labelsize=6)

    H_m_recover = np.fft.ifft(hk_normal_sym)
    ax5.plot(H_m_recover )
    ax5.set_xticks ( x_axis_sym )
    ax5.tick_params(axis='x', labelsize=6)
    ax5.set_title(f'H(m) {C_FFT_N-1} tap ' , fontsize=6)

    dir = 1
    walk = 0

    while True:
        if walk == 1:
            dir = 1
        elif walk == C_FFT_FOLD_N-1:
            dir = -1
        walk += dir 

        # fetch subset of hk points
        hk_dynamic = np.hstack( ( 
            hk_normal[  -(walk + 1)  : -1 ],
            hk_normal[0],
            hk_normal[ 1: walk + 1 ],
        ))

        pad = analysis_window( C_FFT_N-1, np.arange(-walk, walk + 1), ax4 )
        blkman = blackman_window(hk_dynamic,C_FFT_N-1)
        ntaps = hk_dynamic.size 
        hk_dynamic = Chapter2.window.expander(hk_dynamic, C_FFT_N-1)
        hk_dynamic_fft = np.fft.fft(hk_dynamic).__abs__()
        hk_dynamic_blkman = blkman * hk_dynamic
        hk_dynamic_blkman_fft = np.fft.fft(hk_dynamic_blkman).__abs__()
 
        ax6.clear()
        normal_max = hk_dynamic_fft.max()
        blackman_max = hk_dynamic_blkman_fft.max()
        cuur_max = normal_max
        if cuur_max < blackman_max:
            cuur_max = blackman_max
        hk_dynamic_fft_dB = 20 * np.log10(hk_dynamic_fft / cuur_max)
        hk_dynamic_blkman_fft_dB = 20 * np.log10(hk_dynamic_blkman_fft / cuur_max)
        ax6.plot( x_axis_sym, hk_dynamic_fft_dB, label='w/o blackman window')
        ax6.set_xticks ( x_axis_sym )
        ax6.tick_params(axis='x', labelsize=6)
        ax6.legend(fontsize=5, loc=1)
        ax6.set_title(f'H(m) {ntaps} tap ' , fontsize=6)
        
        ax7.clear()
        ax7.plot( x_axis_sym, hk_dynamic_blkman_fft_dB, label='w/ blackman window')
        ax7.set_xticks ( x_axis_sym )
        ax7.tick_params(axis='x', labelsize=6)
        ax7.legend(fontsize=5, loc=1)
        plt.pause(0.2)
        pad.remove()