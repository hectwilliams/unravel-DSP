import matplotlib.pyplot as plt 
import numpy as np 

C_2PI = np.pi * 2.0

# omega (w) frequency
omega = np.linspace(-np.pi, np.pi, 1000)

# Define the polynomial coefficients
numerator_coef = np.complex64 ( [0.0605 , 0.121 , 0.0605] ) 
denominator_ceof = np.complex64 ( [1, -1.194, 0.436]  )

# Create the transfer function
numerator_poly = np.zeros(shape=omega.size,dtype=np.complex128)
denominator_poly = numerator_poly.copy()
for i in range(numerator_coef.__len__()):
    """
    polynomial styles 
        1)  a*z^2 + b*z^1 + c ( polynomial form)
        2)  a*z^0 + b*z^-1 + c*z^-2 (invert polynomial form)

        * style 2 used
        * z replaced with exp(-1j*k*w)
    """
    numerator_poly += numerator_coef[i] * np.exp(-1j*i*omega)

for i in range(denominator_ceof.__len__()):
    denominator_poly += denominator_ceof[i] * np.exp(-1j*i*omega)
freq_response = numerator_poly / denominator_poly

# setting ticks for x-axis 
fig = plt.figure(figsize=(10, 7)) 
fig.subplots_adjust( hspace=0.4 )

ax1 = fig.add_subplot(4,1,1)
ax2 = fig.add_subplot(4,1,2)
ax3 = fig.add_subplot(2,2,3,  projection= 'polar' )
ax4 = fig.add_subplot(2,2,4,  projection= '3d' )

ticks = [-2, -4,  -8, 0, 8, 4, 2]
new_ticks = np.array(list(map(lambda x: x if x ==0 else 1/x , ticks)) )
new_ticks = np.round(new_ticks * np.pi * 2, 3)
ax1.set_xticks( new_ticks  ) 
ax1.tick_params(axis='x', labelsize=6)
ax1.set_xlabel('Freq (radian per second)', fontsize=6)
ax1.set_title('2nd Order IIR Filter Magnitude', fontsize=6)
# twin axes above
ax_twin = ax1.twiny()
ax_twin.set_xlim(ax1.get_xlim())
label = np.array(list(map(lambda x: f'fs/{x}' if x !=0 else '0', ticks)))
ax_twin.set_xticks(ax1.get_xticks())
ax_twin.set_xticklabels(label, fontsize=6)

ax2.set_xticks(new_ticks)
ax2.set_title('2nd Order IIR Filter Phase', fontsize=6)
ax2.tick_params(axis='x', labelsize=6)

ax1.plot(omega, freq_response.__abs__())
ax2.plot(omega, np.angle(freq_response))

# poles/zeros 
zero_roots = np.roots(numerator_coef )
pole_roots =  np.roots(denominator_ceof)
print(pole_roots)
# Show the plot
for data in pole_roots:
    ax3.plot ( np.angle(data)  , np.abs(data)  , marker='x', markersize=3)
for data in zero_roots:
    ax3.plot ( np.angle(data)  , np.abs(data)  , marker='o', markersize=3)

ax3.grid(True)
ax3.set_yticklabels([])
ax3.set_rmax( 1 )

# IIR surface 
real_axis = np.linspace(-1, 1, 1000)
imag_axis =  np.linspace(-1, 1, 1000)
xx, yy = np.meshgrid(real_axis, imag_axis)
z_ = xx + 1j*yy
zz_numerator = np.zeros(shape=xx.shape, dtype=np.complex128)
for i in range(numerator_coef.size-1, -1, -1):
    # use polynomial style 1)
    zz_numerator += numerator_coef[(numerator_coef.size-1) - i] * np.pow(z_, i, dtype=np.complex128)
zz_denominator = np.zeros(shape=xx.shape, dtype=np.complex128)
for i in range(denominator_ceof.size-1, -1, -1):
    zz_denominator += denominator_ceof[  (denominator_ceof.size-1) - i] * np.pow(z_, i, dtype=np.complex128)
zz = np.divide(zz_numerator, zz_denominator)
zz_abs = zz.__abs__() / zz.__abs__().max()
filter_indices =  (zz_abs > 0.101)
zz_abs_trim = filter_indices[filter_indices]
ax4.plot(xx[filter_indices], yy[filter_indices], zz_abs[filter_indices],  color='gray')
[ax4.tick_params(axis=ch, labelsize=3) for ch in list('xyz') ]
ax4.set_xlabel('real' ,fontsize=6)
ax4.set_ylabel('imag' ,fontsize=6)
ax4.set_zlabel('|H(z)|' ,fontsize=6  )
ax4.set_xlim(-1 , 1)
ax4.set_ylim(-1, 1)

# draw stability circle edge 
w = np.linspace(0, C_2PI, 100)
values = np.exp(1j * w)
for c in values:
    ax4.scatter(c.real, c.imag, s=0.5, marker='.', color='gray', alpha=0.8)
plt.show()