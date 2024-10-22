"""Draw unit responses of four processes"""
import matplotlib.pyplot as plt 
import numpy as np 
C_NUM_SAMPLES = 20

fig, ax = plt.subplots(nrows=2, ncols=2)
fig2, ax2 = plt.subplots(nrows=2, ncols=2)

n = np.linspace(-10, 10, C_NUM_SAMPLES)

x_n = np.hstack(  (np.zeros(shape=C_NUM_SAMPLES//2) , np.ones(shape=(1)) , np.zeros(shape=C_NUM_SAMPLES//2)   )) 

if C_NUM_SAMPLES % 2 == 0:
    # event number of samples derive 1 extra sample, reduce by 1
    x_n = x_n[:-1]

def fourth_order_comb (x: np.ndarray, order=4):
    y = np.zeros(shape=(C_NUM_SAMPLES) )
    for i in range(C_NUM_SAMPLES):
        delta = i - order
        x_i = x[i]
        x_i_minus_4 = 0 if delta < 0 else x[delta]
        y[i] =  x_i - x_i_minus_4 
    return y
def integrator(x: np.ndarray):
    y = np.zeros(shape=(C_NUM_SAMPLES) )
    for i in range(C_NUM_SAMPLES):
        x_i = x[i] 
        y_n_minus_1 = 0 if i - 1 < 0  else y[i - 1]
        y[i] = x_i + y_n_minus_1
    return y
def leaky_integrator(x: np.ndarray, constamt_A = 0.5):
    y = np.zeros(shape=(C_NUM_SAMPLES) )
    for i in range(C_NUM_SAMPLES):
        x_i = x[i] 
        y_n_minus_1 = 0 if i - 1 < 0  else y[i - 1]
        y[i] = x_i* constamt_A + y_n_minus_1 * ( 1 - constamt_A)
    return y
def differentitator(x: np.ndarray):
    y = np.zeros(shape=(C_NUM_SAMPLES) )
    for i in range(C_NUM_SAMPLES):
        delta = i - 2
        x_i = x[i]
        x_i_minus_2 = 0 if delta < 0 else x[delta]
        y[i] =  (x_i - x_i_minus_2 ) * 0.5

    return y
if __name__ == '__main__':
    y_n = fourth_order_comb(x_n)
    ax[0,0].vlines( x = n , ymin= np.zeros(shape=C_NUM_SAMPLES),  ymax=x_n, alpha=0.5 )
    ax[0,0].stem(n , y_n, '-.')
    ax[0,0].set_title('h(n) fourth order comb system')
    ax[0,0].set_xlabel('n')
    ax[0,0].set_ylabel('impulse response h(n)')
    ax2[0,0].step(n, np.cumsum(y_n)) 

    y_n = integrator(x_n)
    ax[0,1].vlines( x = n , ymin= np.zeros(shape=C_NUM_SAMPLES),  ymax=x_n, alpha=0.5 )
    ax[0,1].stem(n , y_n, '-.')
    ax[0,1].set_title('h(n) integrator system')
    ax[0,1].set_xlabel('n')
    ax[0,1].set_ylabel('impulse response h(n)')
    ax2[0,1].step(n, np.cumsum(y_n)) 

    y_n = leaky_integrator(x_n)
    ax[1,0].vlines( x = n , ymin= np.zeros(shape=C_NUM_SAMPLES),  ymax=x_n, alpha=0.5 )
    ax[1,0].stem(n , y_n, '-.')
    ax[1,0].set_title('h(n) Leaky integrator system')
    ax[1,0].set_xlabel('n')
    ax[1,0].set_ylabel('impulse response h(n)')
    ax2[1,0].step(n, np.cumsum(y_n)) 

    y_n = differentitator(x_n)
    ax[1,1].vlines( x = n , ymin= np.zeros(shape=C_NUM_SAMPLES),  ymax=x_n, alpha=0.5 )
    ax[1,1].stem(n , y_n, '-.')
    ax[1,1].set_title('h(n) Differentiator system')
    ax[1,1].set_xlabel('n')
    ax[1,1].set_ylabel('impulse response h(n)')
    ax2[1,1].step(n, np.cumsum(y_n)) 

    fig2.suptitle('step response')
    plt.show()



