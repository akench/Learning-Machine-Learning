import matplotlib.pyplot as plt
from scipy.io import wavfile
import os.path as path
import io
from PIL import Image
import librosa
import librosa.display
import numpy as np

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
        if abs(data[silence_index]) > almost_zero:
            break

    data = data[silence_index : ]

    #removes ending silence
    silence_index = len(data) - 1
    for silence_index in reversed(range(len(data))):
        if abs(data[silence_index]) > almost_zero:
            break
    data = data[ : silence_index + 1]

    return data


def replace_silence(data):

    for i in range(len(data)):
        if abs(data[i]) <= almost_zero:
            data[i] = 1

    return data


def get_wav_info(wav_file):
    rate, data = wavfile.read(wav_file)

    if len(data.shape) > 1:
        data = data.sum(axis = 1) / 2
    return rate, data


def graph_spectrogram(wav_file, secs_per_spec = 10):

    vid_id = wav_file.split('/')[-1].split('.')[0]
    emot = wav_file.split('/')[1]

    save_path = 'gen_specs/' + emot + '/' + vid_id

    #if the current .wav has already been converted, skip
    if path.exists(save_path + '_0.png'):
        print('exists')
        return


    rate, data = get_wav_info(wav_file)
    print('rate: ', rate)



    data = remove_trailing_silence(data)

    #replacing silence causes some weird spectrograms
    # data = replace_silence(data)
    split_data = split_list_by_num_samples(data, rate * secs_per_spec)

    # Length of the windowing segments
    nfft = 256
    sampling_freq = 256


    name_index = 0
    for mini_data in split_data:
        # pxx, freqs, bins, im = plt.specgram(mini_data, NFFT = nfft, Fs = sampling_freq, scale = 'dB', cmap = 'Greys')
        # plt.axis('off')

        # buf = io.BytesIO()
        # plt.savefig(buf,
        #             format='jpg',
        #             dpi=300, #size of image
        #             frameon='false',
        #             aspect='equal',
        #             bbox_inches='tight',
        #             pad_inches=-.2)
        # buf.seek(0)
        # img = Image.open(buf)
        # img = img.crop((49, 0, 1480, 1050))
        # img.save(save_path + str(name_index) + '.jpg')
        # buf.close()
        # plt.clf()

        S = librosa.feature.melspectrogram(mini_data, sr=rate, n_mels=128)
        log_S = librosa.logamplitude(S, ref_power=np.max)

        plt.figure(figsize=(secs_per_spec, 7))

        librosa.display.specshow(log_S, sr=rate, x_axis='time', y_axis='mel')

        # plt.colorbar(format='%+02.0f dB')

        plt.tight_layout()
        plt.axis('off')
        plt.draw()

        import io
        buf = io.BytesIO()
        plt.savefig(buf, bbox_inches='tight', pad_inches = 0, dpi=50)

        buf.seek(0)
        img = Image.open(buf)
        img = img.crop((120, 4, img.size[0] - 3, img.size[1] - 20))
        img = img.convert('L')
        img.save(save_path + '_' + str(name_index) + '.jpg')
        buf.close()

        plt.clf()
        plt.cla()
        plt.close()

        name_index += 1




def all_wavs_to_spec(wav_dir):
    import glob
    wavs = glob.glob(wav_dir + '/*.wav')

    for w in wavs:
        graph_spectrogram(w)


