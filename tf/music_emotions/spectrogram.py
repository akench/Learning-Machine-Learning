import matplotlib.pyplot as plt
from scipy.io import wavfile


def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    mono_data = data.sum(axis = 1) / 2
    return rate, mono_data

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    print('RATE!!!', rate)

    #removes beginning silence
    silence_index = 0
    for silence_index in xrange(len(data)):
        if data[silence_index] != 0:
            break

    data = data[silence_index : ]

    #removes ending silence
    silence_index = len(data) - 1
    for silence_index in reversed(xrange(len(data))):
        if data[silence_index] != 0:
            break
    data = data[ : silence_index + 1]



    # Length of the windowing segments
    nfft = 256
    # Sampling frequency
    fs = 2560   

    print(len(data))
    data = data[int(5339044 / 3): int(5339044 / 2)]


    pxx, freqs, bins, im = plt.specgram(data, nfft,fs)
    plt.axis('off')
    plt.savefig('test.png',
                dpi=100, #resolution of image
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=1) 



graph_spectrogram('davy_dl.wav')