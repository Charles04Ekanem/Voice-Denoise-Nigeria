import matplotlib.pyplot as plt
from scipy.io import wavfile

filepath = r'C:\Users\CHARLES EKANEM\Documents\ICMEAS\data\clean_train\clean_00000.wav'

sample_rate, data = wavfile.read(filepath)

duration = len(data) / sample_rate
time = [i / sample_rate for i in range(len(data))]

plt.figure(figsize=(10, 4))
plt.plot(time, data, linewidth=0.7)
plt.title(f"Waveform of {filepath}")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.tight_layout()
plt.show()