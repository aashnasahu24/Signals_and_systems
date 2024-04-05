import numpy as np
import matplotlib.pyplot as plt
from scipy import signal



#sampling frequency from the audio file
sampl_freq=44100
T = 1.0/sampl_freq
#order of the filter
order=4

#cutoff frquency 
cutoff_freq=1000.0 

#digital frequency
Wn=2*cutoff_freq/sampl_freq  


# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order, Wn, 'low') 

#Bilinear Transform
def H(s):
	num = np.polyval(b,((1+s*(T/2))/(1-s*(T/2)))**(-1))
	den = np.polyval(a,((1+s*(T/2))/(1-s*(T/2)))**(-1))
	H = num/den
	return H
		
#Input and Output
analog_f = np.arange(0,5000,100)

plt.plot(analog_f, (abs(H(1j*analog_f))))
plt.xlabel('$\omega$')
plt.ylabel('$|H(e^{jmath\omega})| $')
plt.grid()
plt.savefig("Butterworth_analog.png")