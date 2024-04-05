#include <stdio.h>
#include <stdlib.h>
#include <complex.h>
#include <time.h>
#include <math.h>

#define EPS 1e-6
#define Pi 3.141592653589
// Function for FFT
complex *custom_fft(int n, complex *a) {
    if (n == 1) return a;
	complex *g = (complex *)malloc(n/2*sizeof(complex));
	complex *h = (complex *)malloc(n/2*sizeof(complex));
	for (int i = 0; i < n; i++) { 
		if (i%2) h[i/2] = a[i];
		else g[i/2] = a[i];
	}
	g = custom_fft(n/2, g);
	h = custom_fft(n/2, h);
	for (int i = 0; i < n; i++) a[i] = g[i%(n/2)] + cexp(-I*2*Pi*i/n)*h[i%(n/2)];
	free(g); free(h);
	return a;
}

// Function for IFFT
complex *custom_ifft(int n, complex *a) {
    if (n == 1) return a;
	complex *g = (complex *)malloc(n/2*sizeof(complex));
	complex *h = (complex *)malloc(n/2*sizeof(complex));
	for (int i = 0; i < n; i++) { 
		if (i%2) h[i/2] = a[i];
		else g[i/2] = a[i];
	}
	g = custom_ifft(n/2, g);
	h = custom_ifft(n/2, h);
	for (int i = 0; i < n; i++) {
		a[i] = g[i%(n/2)] + cexp(I*2*Pi*i/n)*h[i%(n/2)];
		a[i] /= 2;
	}
	free(g); free(h);
	return a;
}

// Function for Convolution
complex *custom_convolve(complex *h, complex *x, int n) {
    complex *a = (complex *)calloc(n, sizeof(complex));
	for (int i = 0; i < n; i++) for (int j = 0; j <= i; j++) a[i] += h[j]*x[i - j];
	return a;
}

int main() {
    FILE *f_time_fft = fopen("time_fft.txt", "w");
    FILE *f_time_conv = fopen("time_conv.txt", "w");

    for (int j = 0; j <= 15; j++) {
        srand(time(0));
        int n = 1 << j;
        complex *x = (complex *)malloc(sizeof(complex) * n);
        complex *h = (complex *)malloc(sizeof(complex) * n);

        for (int i = 0; i < n; i++) {
            x[i] = (double)rand() / RAND_MAX;
            h[i] = (double)rand() / RAND_MAX;
        }

        // Measure time for FFT
        clock_t fft_begin = clock();
        custom_fft(n, x);
        clock_t fft_end = clock();

        // Measure time for IFFT
        clock_t ifft_begin = clock();
        custom_ifft(n, x);
        clock_t ifft_end = clock();

        // Measure time for convolution
        clock_t conv_begin = clock();
        custom_convolve(h, x, n);
        clock_t conv_end = clock();

        // Calculate total time for FFT and IFFT
        double time_fft = (double)(fft_end - fft_begin) / CLOCKS_PER_SEC;
        double time_ifft = (double)(ifft_end - ifft_begin) / CLOCKS_PER_SEC;
        double total_time_fft = time_fft + time_ifft;

        // Write total time for FFT and IFFT to file
        fprintf(f_time_fft, "%lf\n", total_time_fft);

        // Write time for convolution to file
        double time_conv = (double)(conv_end - conv_begin) / CLOCKS_PER_SEC;
        fprintf(f_time_conv, "%lf\n", time_conv);

        free(x);
        free(h);
    }

    fclose(f_time_fft);
    fclose(f_time_conv);

    return 0;
}
