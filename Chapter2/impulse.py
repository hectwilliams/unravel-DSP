import numpy as np 
import matplotlib.pyplot as plt 
C_SSB_N = 8
C_N = C_SSB_N * 2
if __name__ == '__main__':
    fig = plt.figure()
    ax1 = fig.add_subplot(3, 1, 1)
    ax2 = fig.add_subplot(3, 1, 2)
    ax3 = fig.add_subplot(3, 1, 3)

    y1 = np.zeros(shape=C_N)
    y2 = y1.copy()

    y1[0] = 1
    y2[1] = 1 
    y2[0] = 1 

    y1_fft =np.abs(np.fft.fft(y1))
    y2_fft = np.abs(np.fft.fft(y2))
    x = np.arange(0, C_N)
    ax1.vlines(x, np.zeros(shape=C_N), y1, color='black')
    ax1.vlines(x, np.zeros(shape=C_N), y2, color='blue')
    ax2.plot(y1_fft, color='black')
    ax3.plot(y2_fft, color='blue')
    plt.show()


