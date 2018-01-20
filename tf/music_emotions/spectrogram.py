import matplotlib.pyplot as plt
from scipy.io import wavfile


almost_zero = 0.001

def split_list_by_num_samples(data, num_samples):

    new = []
    index = 0
    while index + num_samples < len(data):

        new.append(data[index : index + num_samples])
        index += num_samples

    return new

def remove_trailing_silence(data):
    #removes beginning silence
    silence_index = 0
    for silence_index in range(len(data)):
        if data[silence_index] > almost_zero:
            break

    data = data[silence_index : ]

    #removes ending silence
    silence_index = len(data) - 1
    for silence_index in reversed(range(len(data))):
        if data[silence_index] > almost_zero:
            break
    data = data[ : silence_index + 1]

    return data




def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)

    if len(data.shape) > 1:
        data = data.sum(axis = 1) / 2
    return rate, data

def graph_spectrogram(wav_file):
    rate, data = get_wav_info(wav_file)
    print('RATE!!!', rate)

    print(len(data) / rate)
    data = remove_trailing_silence(data)
    print(len(data) / rate)
    # quit()
    split_data = split_list_by_num_samples(data, rate * 10)
    # split_data = []
    # split_data.append(data)

    # Length of the windowing segments
    nfft = 256  
    sampling_freq = 2


    name_index = 0
    for mini_data in split_data:
        pxx, freqs, bins, im = plt.specgram(mini_data, NFFT = nfft, Fs = sampling_freq, scale = 'dB')
        plt.axis('off')
        plt.savefig('testdir/spec' + str(name_index) + '.png',
                    dpi=300, #size of image
                    frameon='false',
                    aspect='equal',
                    bbox_inches='tight',
                    pad_inches=-.2) 
        plt.clf()

        name_index += 1



graph_spectrogram('dl/output.wav')












# import matplotlib.pyplot as plt
# from scipy.io import wavfile

# def split_list_by_num_samples(data, num_samples):

#     new = []
#     index = 0
#     while index + num_samples < len(data):

#         new.append(data[index : index + num_samples])
#         index += num_samples

#     return new

# def remove_trailing_silence(data):
#     #removes beginning silence
#     silence_index = 0
#     for silence_index in range(len(data)):
#         if data[silence_index] != 0:
#             break

#     data = data[silence_index : ]

#     #removes ending silence
#     silence_index = len(data) - 1
#     for silence_index in reversed(range(len(data))):
#         if data[silence_index] != 0:
#             break
#     data = data[ : silence_index + 1]

#     return data

# def replace_silence(data):

#     for i in range(len(data)):
#         if data[i] < 0.001:
#             data[i] = 1
#     return data



# def get_wav_info(wav_file):
#     rate, data = wavfile.read(wav_file)

#     if len(data.shape) > 1:
#         data = data.sum(axis = 1) / 2
#     return rate, data

# def graph_spectrogram(wav_file):
#     rate, data = get_wav_info(wav_file)
#     print('RATE!!!', rate)

#     print(len(data) / rate)

#     data = remove_trailing_silence(data)
#     data = replace_silence(data)

#     print(len(data) / rate)

#     split_data = split_list_by_num_samples(data, rate * 10)

#     # Length of the windowing segments
#     nfft = 256  
#     sampling_freq = 256  


#     name_index = 0
#     for mini_data in split_data:
#         pxx, freqs, bins, im = plt.specgram(mini_data, NFFT = nfft, Fs = sampling_freq, scale = 'dB')
#         plt.axis('off')
#         plt.savefig('testdir/spec' + str(name_index) + '.png',
#                     dpi=300, #size of image
#                     frameon='false',
#                     aspect='equal',
#                     bbox_inches='tight',
#                     pad_inches=-.2) 
#         plt.clf()

#         name_index += 1



# graph_spectrogram('bye.wav')