import ctypes
import numpy as np

# Load the shared library
fft_lib = ctypes.CDLL('fft.so')

# Define the complex number structure
class Complex(ctypes.Structure):
    _fields_ = [('real', ctypes.c_double),
                ('imag', ctypes.c_double)]

# Define the signature of the C functions
fft_lib.fft.argtypes = [ctypes.POINTER(Complex), ctypes.c_int]

# Helper function to convert Python complex array to C complex array
def py_complex_to_c_array(arr):
    c_arr = (Complex * len(arr))()
    for i, val in enumerate(arr):
        c_arr[i].real = val.real
        c_arr[i].imag = val.imag
    return c_arr

# Helper function to convert C complex array to Python complex array
def c_array_to_py_complex(arr, length):
    py_arr = []
    for i in range(length):
        py_arr.append(complex(arr[i].real, arr[i].imag))
    return py_arr

# Define the input array
input_array = np.array([1, 1, 1, 1, -1, -1, -1, -1], dtype=np.complex128)

# Call the C function
input_array_c = py_complex_to_c_array(input_array)
fft_lib.fft(input_array_c, ctypes.c_int(len(input_array)))
output_array = c_array_to_py_complex(input_array_c, len(input_array))

# Display the result
print("FFT result:", output_array)
