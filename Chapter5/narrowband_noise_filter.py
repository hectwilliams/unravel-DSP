import matplotlib.pyplot as plt 
import numpy as np 
C_2PI = 2 * np.pi

if __name__ == '__main__':
    fig = plt.figure() 
    fig.subplots_adjust( hspace=0.8 )
    ax1 = fig.add_subplot(2,1,1)
    ax2 = fig.add_subplot(2,1,2)

w = np.linspace(-np.pi , np.pi , 10000 )

# Filter Freq Response 
h = [1, 1.61 + 5.9e-3*1j, 1] # h[1] controls location of notch refer to page 246 in 'that' DSP textbook
freq_resp =  np.zeros( shape=w.shape)
for i in range(3):
    freq_resp =  freq_resp + ( h[i] * np.exp(-1j * i *  w) )

indices = np.nonzero(  np.abs(freq_resp)  < 0.006)  
print(indices)

Hw_abs = np.abs(freq_resp)
Hw_norm = Hw_abs/ Hw_abs.max()
Hw_db  = 20 * np.log(Hw_norm) 
ax1.plot(w , Hw_db, label='response')
ax1.vlines(w[1012], Hw_db.min(), 3, color='red', label='notch')
ax1.vlines(w[8987], Hw_db.min(), 3, color='red')
ax1.set_title(f' Narrowband Noise Reduce Filter: Freq Reponse', fontsize=5)    
ax1.legend(fontsize=5, loc=1)
ax1.set_xlabel('radians/sample')
ax1.set_xlabel('radians/sample')
ax2.plot(w, np.angle(freq_resp))
ax2.set_title(f' Narrowband Noise Reduce Filter: Phase Reponse', fontsize=5)    
plt.show()
