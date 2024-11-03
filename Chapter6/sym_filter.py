import matplotlib.pyplot as plt 
import numpy as np 

fig = plt.figure(figsize=(10, 7)) 
ax1 = fig.add_subplot(1,1,1,  projection= 'polar' )

phases = np.array( [ (m*2 *np.pi)/5 for m in range(5)])
data = np.zeros(shape=(5,5))

for i in range(5):
    phase = phases[i]
    for n in range(5):
        data[i][n] = ( n * phase)


Hm_1 = np.exp(data[1] * -1j )
for a in Hm_1:
    ax1.plot( np.angle(a) , np.abs(a) , marker='o', color='pink', markersize=3)
ax1.set_yticklabels([])

Hm_4 = np.exp(data[4] * -1j )
for a in Hm_4:
    ax1.plot( np.angle(a) , np.abs(a) , marker='o', color='blue', markersize=3, alpha=0.4)
ax1.set_yticklabels([])

"""
    Frequency Response at index 1 and index 4 are equal because their phases used for the point to point DFT of h(k) are equal. 

    H_1 Phase  --> [   0.   72.  144.  216.  288 ]
    H_1 Phase  --> [   0.   1*72.  2*72.  3*72 .  4*72 ]
        
    
    H_4 Phase  --> [   0.  288.  576.  864. 1152 ]
    H_4 Phase  -->  [   0.  4*72. 8*72. 12*72. 16*72 ]
    H_4 Phase(alias)  --> [   0.  288.  216.  144.  72 ]

    If impulse reponse is symmetric  so H_1 and H_4 produce equal magnitudes 
    Overall symmetic magnitude and linear phase are created along the passband
"""
plt.show() 
