from __future__ import unicode_literals
import youtube_dl

outfile = 'dl/output.%(ext)s'

ydl_opts = {
    'format': 'bestaudio/best',
    'postprocessors': [{
        'key': 'FFmpegExtractAudio',
        'preferredcodec': 'wav',
        'preferredquality': '192',
    }],
    'outtmpl': outfile
}
with youtube_dl.YoutubeDL(ydl_opts) as ydl:
	link = 'https://www.youtube.com/watch?v=YyknBTm_YyM'
	ydl.download([link])
