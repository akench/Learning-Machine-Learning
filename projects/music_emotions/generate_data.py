from __future__ import unicode_literals
import youtube_dl
from random import *
import os.path
from make_spectrogram import all_audio_to_spec

def dl_audio(url, emot, save = True):

	vid_id = url.split('=')[1]

	if emot is None:
		outfile = vid_id
	else:	
		outfile = 'audio/' + emot + '/' + vid_id



	if os.path.exists(outfile + '.mp3'):
		print('.', end='', flush=True)
		return

	ydl_opts = {
	    'format': 'bestaudio/best',
	    'postprocessors': [{
	        'key': 'FFmpegExtractAudio',
	        'preferredcodec': 'mp3',
	        'preferredquality': '192',
	    }],
	    'outtmpl': outfile + '.%(ext)s'
	}
	with youtube_dl.YoutubeDL(ydl_opts) as ydl:
		ydl.download([url])

	return outfile



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
	print('')

	import glob
	dirs = glob.glob('audio/*')

	for d in dirs:

		spec_path = 'gen_specs/' + d.split('/')[1]
		if not os.path.exists(spec_path):
			os.mkdir(spec_path)

		all_audio_to_spec(d)


