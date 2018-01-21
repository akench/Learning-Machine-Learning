import matplotlib.pyplot as plt
from scipy.io import wavfile
import numpy as np

def remove_trailing_silence(data):
    #removes beginning silence
    silence_index = 0
    for silence_index in range(len(data)):
        if data[silence_index] == 0:
            break

    data = data[silence_index : ]

    #removes ending silence
    silence_index = len(data) - 1
    for silence_index in reversed(range(len(data))):
        if data[silence_index] == 0:
            break
    data = data[ : silence_index + 1]

    return data


def replace_silence(data):

    for i in range(len(data)):
        if data[i] == 0:
            data[i] = 1
        # data[i] += 1

    return data

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    nfft = 256  # Length of the windowing segments
    fs = 256    # Sampling frequency
    print(len(data))

    data = remove_trailing_silence(data)
    data = replace_silence(data)
    pxx, freqs, bins, im = plt.specgram(data, nfft,fs)
    plt.axis('off')
    plt.savefig('testspec.png',
                dpi=100, # Dots per inch
                frameon='false',
                aspect='normal',
                bbox_inches='tight',
                pad_inches=0) # Spectrogram saved as a .png

def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)
    if len(data.shape) > 1:
        data = data.sum(axis = 1) / 2
    print(data.shape)
    print('max', data[np.argmax(data)])
    print('min', data[np.argmin(data)])


    return rate, data

if __name__ == '__main__': # Main function
    wav_file = 'wav_files/tense/3X9LvC9WkkQ.wav' # Filename of the wav file
    graph_spectrogram(wav_file)
