
import matplotlib.pyplot as plt 
import numpy as np 

C_2PI = 2 * np.pi
N = 8

if __name__ == '__main__':
    fig = plt.figure() 
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

    freq_axis = np.arange(0, N)
    w = np.linspace(0, C_2PI, 1000)
    h = np.zeros(shape=N)
    h[ [0, N-1] ] =1
    # h = np.array ( [1,0,0,0,0,0,0,1] )
    H_w = np.zeros(shape=w.shape)
    for k in freq_axis:
        H_w = H_w + ( h[k] * np.exp( -1j * k * w ) )
    H_w_mag = H_w.__abs__()
    H_w_norm = H_w_mag / H_w_mag.max()
    H_w_db = 20 * np.log10(H_w_norm)

    ax1.scatter(freq_axis, h, s=4)
    ax1.vlines(freq_axis, np.zeros(shape=h.shape), h)
    ax1.set_title('Time-Domain Impulse Response N = 8', fontsize=6)
    ax2.plot(w, H_w_db)
    ax2.set_title(f'Digital Reverberator Frequency Magnitude Response' , fontsize=6)
    plt.show()