# Learning Machine Learning
Tensorflow and ML stuff

## Projects

### Music Emotions

Summary: Organizes your music library depending on the emotion that the song evokes
Can give a breakdown of the emotions (by %) present in a given song (either by mp3 file or youtube link)

Categories : Happy, Sad, Motivational, Relaxing, Angry, Tense

Use Case:  If a person is feeling excited about something, they can simply navigate to the "Motivational" directory in their organized song library and listen to the songs there. People enjoy listening to music that matches their current emotions.

How it was done:
* The mp3 file was decoded and split up into segments of 10 seconds each
* Each audio segment was converted to a spectrogram using the Librosa library
* Spectrogram of sad song segment
![alt text](projects/music_emotions/readme_pics/spec_pic.jpg "spectrogram of sad song")

