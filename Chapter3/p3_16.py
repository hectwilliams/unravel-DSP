import numpy as np 
import matplotlib.pyplot as plt 
C_2PI = np.pi * 2
C_N = 16 
C_RANGE = np.arange(0, C_N )
C_UNIQUE_RANGE = np.arange(0, np.int32(C_N//2) + 1 )
K = 2
h1 =  (1 - np.cos(C_2PI*K*C_RANGE*np.divide(1, C_N))  )
h1_fft = np.fft.fft(h1)
h1_fft_mag = np.abs(h1_fft)
fig = plt.figure()
ax1 = fig.add_subplot(3, 1, 1)
ax2 = fig.add_subplot(3, 1, 2)
ax3 = fig.add_subplot(3, 1, 3)

m0_analysis = np.exp(1j * C_2PI * 0 * np.divide(1, C_N) * C_RANGE)
c0 = m0_analysis * h1
m1_analysis = np.exp(1j * C_2PI * 1 * np.divide(1, C_N) * C_RANGE)
c1 = m1_analysis * h1
c1_dot = np.dot(m1_analysis, h1) 
c0_dot = np.dot(m0_analysis, h1) 

print(c0_dot.__abs__(), c1_dot.__abs__())

# c1_dot_mag = np.abs(c1_dot)

m1_analysis = np.exp(1j * C_2PI * 1 * np.divide(1, C_N) * C_RANGE)
ax1.scatter(C_RANGE, h1[C_RANGE], color='gray', alpha=0.5)
ax1.scatter(C_RANGE, c0[C_RANGE], color='yellow', s=0.5)
ax2.scatter(C_RANGE, h1[C_RANGE], color='gray', alpha=0.5)
ax2.scatter(C_RANGE, c1[C_RANGE], color='yellow', s=0.5)
# ax2.plot(C_RANGE, m1_analysis.real[C_RANGE], linestyle='dotted', color='black', linewidth=0.5)
# ax2.plot(C_RANGE, m1_analysis.imag[C_RANGE], linestyle = 'dashed', color='black', linewidth=0.5)

# ax1.scatter(C_RANGE, m1_analysis[C_RANGE], color='red', alpha=0.5)

# ax1.vlines(C_RANGE, np.zeros(shape=C_RANGE.size), m1_analysis[C_RANGE], linewidth=0.5)
ax3.vlines(C_UNIQUE_RANGE, np.zeros(shape=C_UNIQUE_RANGE.size), h1_fft_mag[C_UNIQUE_RANGE])
plt.show()

"""

[0.         0.03806023 0.14644661 0.30865828 0.5        0.69134172
 0.85355339 0.96193977 1.         0.96193977 0.85355339 0.69134172
 0.5        0.30865828 0.14644661 0.03806023]

 """