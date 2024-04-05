import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
import soundfile as sf

#read .wav file 
input_signal, fs = sf.read('soundfile.wav') 

#sampling frequency 
sampl_freq = fs

#order of the filter
order = 4

#cutoff frequency 
cutoff_freq = 1000.0 

#digital frequency
Wn = 2 * cutoff_freq / sampl_freq  

# b and a are numerator and denominator polynomials respectively
b, a = signal.butter(order, Wn, 'low') 

zeros, poles, gain = signal.tf2zpk(b, a)

plt.figure(figsize=(10, 8))
plt.scatter(zeros.real, zeros.imag, marker='o', color='b', label='Zeros', s=100)
plt.scatter(poles.real, poles.imag, marker='x', color='r', label='Poles', s=100)
plt.axvline(0, color='k', linestyle='--', linewidth=0.5)
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('Real Part')
plt.ylabel('Imaginary Part')
plt.title('Pole-Zero Plot')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.7)
plt.gca().spines['top'].set_color('none')
plt.gca().spines['right'].set_color('none')
plt.gca().spines['bottom'].set_linewidth(2)
plt.gca().spines['left'].set_linewidth(2)
plt.tick_params(axis='both', which='major', labelsize=12)
plt.tick_params(axis='both', which='minor', labelsize=10)
plt.tight_layout()
plt.savefig("../figs/Pole_Zero_Plt.png")
