% Matlab Code for plotting analog domain butterworth filter 


fc = 1000;
fs = 44100;

[b, a] = butter(4, fc/(fs/2));

% Calculate frequency response
[H, freq] = freqz(b, a, [], fs);

plot(freq, 20*log10(abs(H)));
xlabel('Frequency (Hz)');
ylabel('Magnitude (dB)');
title('Magnitude Response');
ylim([-100 20]);
grid on;

% Draw a vertical dotted line at the cutoff frequency (1000 Hz)
hold on;
xline(fc, '--', 'Color', 'r');
legend('Magnitude Response', 'Cutoff Frequency (1000 Hz)', 'Location', 'best');
hold off;
