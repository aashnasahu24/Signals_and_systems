import numpy as np
import matplotlib.pyplot as plt

# Read data from time_fft.txt
conv_times = np.loadtxt('conv.txt')

# Read data from time_conv.txt
fft_times = np.loadtxt('fft.txt')


sizes = np.array([2**i for i in range(16)])

# Plotting
plt.stem(sizes, fft_times, 'b',label='FFT + IFFT Time')
plt.stem(sizes, conv_times,'r', label='Convolution Time')
plt.xlabel('Input Size (n)')
plt.ylabel('Time (seconds)')
plt.title('Time Complexity Comparison')
plt.xscale('log', base=2)  # Logarithmic scale for x-axis
plt.yscale('log')  # Logarithmic scale for y-axis
plt.legend()
plt.grid(True)
plt.savefig("Time Complexity Comparision.png")
