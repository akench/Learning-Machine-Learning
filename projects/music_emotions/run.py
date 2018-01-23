from __future__ import unicode_literals
import youtube_dl
from random import *
import os.path as path
from spectrogram import all_wavs_to_spec

def dl_audio(url, emot):

	vid_id = url.split('=')[1]
	outfile = 'wav_files/' + emot + '/' + vid_id 



	if path.exists(outfile + '.wav'):
		print('already exists!')
		return

	ydl_opts = {
	    'format': 'bestaudio/best',
	    'postprocessors': [{
	        'key': 'FFmpegExtractAudio',
	        'preferredcodec': 'wav',
	        'preferredquality': '192',
	    }],
	    'outtmpl': outfile + '.%(ext)s'
	}
	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		ydl.download([url])



def read_file():

	with open('songs.txt', 'r') as f:
		for line in f:
			line = line.rstrip()
			if 'http' in line:
				dl_audio(url = line, emot = curr_emot)
			elif line.strip() != '':
				curr_emot = line


def main():
	read_file()

	# import glob
	# dirs = glob.glob('wav_files/*')
	# for d in dirs:
	# 	all_wavs_to_spec(d)


# main()
dl_audio('https://www.youtube.com/watch?v=l6Gl7AAnT8U', 'test')
