import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from scipy.io import wavfile

# y, sr = librosa.core.load("hungary.wav")
sr, y = wavfile.read("hungary.wav")

if len(y.shape) > 1:
    y = y.sum(axis = 1) / 2


start = 150
duration = 150
y = y[sr * start: sr * (start + duration)]



S = librosa.feature.melspectrogram(y, sr=sr, n_mels=128)
log_S = librosa.logamplitude(S, ref_power=np.max)

plt.figure(figsize=(duration, 5))

librosa.display.specshow(log_S, sr=sr, x_axis='time', y_axis='mel')

# plt.colorbar(format='%+02.0f dB')

plt.tight_layout()
plt.axis('off')
plt.draw()

import io
buf = io.BytesIO()
plt.savefig(buf, bbox_inches='tight', pad_inches = 0, dpi=200)

from PIL import Image
buf.seek(0)
img = Image.open(buf)
img = img.crop((120, 2, img.size[0], img.size[1] - 85))
img.save('none.jpg')
buf.close()