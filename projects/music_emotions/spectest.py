import matplotlib.pyplot as plt
import matplotlib
from scipy.io import wavfile
import numpy as np
from scipy.fftpack import rfft, fftshift

def remove_trailing_silence(data):
    #removes beginning silence
    silence_index = 0
    for silence_index in range(len(data)):
        if abs(data[silence_index]) <= 0.001:
            break

    data = data[silence_index : ]

    #removes ending silence
    silence_index = len(data) - 1
    for silence_index in reversed(range(len(data))):
        if abs(data[silence_index]) <= 0.001:
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
    # data = replace_silence(data)
    pxx, freqs, bins, im = plt.specgram(data[rate*145 :rate*155], nfft,fs)
    # plt.axis('off')
    plt.colorbar()
    plt.savefig(wav_file[:-4] + 'old.png',
                dpi=300, # Dots per inch
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



    # new = []
    # for i in range(len(data)):
    #     if i%2 == 0:
    #         new.append(data[i])
    return rate, data



import os
import wave

import pylab
def spec_test(wav_file):
    sound_info, frame_rate = get_wav_info_test(wav_file)
    pylab.figure(num=None, figsize=(19, 12))
    pylab.subplot(111)
    pylab.title('spectrogram of %r' % wav_file)

    pylab.specgram(sound_info[frame_rate*0 : frame_rate*10], Fs=frame_rate)

    # pylab.ion()
    pylab.savefig(wav_file[: -4] + '.png')


def get_wav_info_test(wav_file):
    wav = wave.open(wav_file, 'r')
    frames = wav.readframes(-1)
    sound_info = pylab.fromstring(frames, 'int16')
    frame_rate = wav.getframerate()
    wav.close()

    print('shape',sound_info.shape)

    if len(sound_info.shape) > 1:
        sound_info = sound_info.sum(axis = 1) / 2

    print('rate:', frame_rate)
    sound_info = remove_trailing_silence(sound_info)
    # sound_info = replace_silence(sound_info)
    return sound_info, frame_rate




if __name__ == '__main__': # Main function
    # wav_file = '/home/super/Desktop/loudclap.wav' # Filename of the wav file
    wav_file = 'wav_files/relaxing/kLp_Hh6DKWc.wav'
    spec_test(wav_file)
    print('done')
    # graph_spectrogram(wav_file)
    print('done')
