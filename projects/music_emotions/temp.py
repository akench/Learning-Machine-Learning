import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

time1 = np.arange(0,5,0.0001)
time = np.arange(0,15,0.0001)
data1=np.sin(2*np.pi*300*time1)
data2=np.sin(2*np.pi*600*time1)
data3=np.sin(2*np.pi*900*time1)
data=np.append(data1,data2 )
data=np.append(data,data3)
print(len(time))
print(len(data))

print(data)

NFFT = 200     # the length of the windowing segments
Fs = 500  # the sampling rate

# plot signal and spectrogram

ax1 = plt.subplot(211)
plt.plot(time,data)   # for this one has to either undersample or zoom in 
plt.xlim([0,15])
plt.subplot(212 )  # don't share the axis
Pxx, freqs, bins, im = plt.specgram(data, NFFT=NFFT,   Fs=Fs,noverlap=100, cmap=plt.cm.gist_heat)
plt.show() 