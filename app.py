import streamlit as st
'''import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten'''
import pydub
from pathlib import Path
import os

curr_dir = os.getcwd()
genres_list = ['blues', 'classical', 'country', 'disco', 'hiphop', 'jazz', 'metal', 'pop', 'reggae', 'rock']

'''def create_model():

  Model = Sequential()
  Model.add(Conv2D(16, (3,3), activation='relu', input_shape=(160, 2049, 1)))
  Model.add(Conv2D(16, (3,3), activation='relu'))
  Model.add(Flatten())
  #Model.add(Dense(100, activation='relu'))
  Model.add(Dense(50, activation='relu'))
  Model.add(Dense(10, activation='sigmoid'))

  return Model

def load_wav(filename):
    # Load encoded wav file
    file_contents = tf.io.read_file(filename)
    # Decode wav (tensors by channels)
    wav, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
    # Removes trailing axis
    wav = tf.squeeze(wav, axis=-1)
    sample_rate = tf.cast(sample_rate, dtype=tf.int64)
    # Goes from 44100Hz to 16000hz - amplitude of the audio signal
    wav = tfio.audio.resample(wav, rate_in=sample_rate, rate_out=16000)
    return wav

def preprocess(file_path):
    wav = load_wav(file_path)
    wav = wav[:480000]
    zero_padding = tf.zeros([480000] - tf.shape(wav), dtype=tf.float32)
    wav = tf.concat([zero_padding, wav],0)
    spectrogram = tf.signal.stft(wav, frame_length=3000, frame_step=3000)
    spectrogram = tf.abs(spectrogram)
    spectrogram = tf.expand_dims(spectrogram, axis=2)
    return spectrogram'''

def upload_and_save_wavfiles():
    uploaded_files = st.file_uploader("upload", type=['wav', 'mp3'], accept_multiple_files=True)
    save_paths = []
    for uploaded_file in uploaded_files:
        if uploaded_file is not None:
            if uploaded_file.name.endswith('wav'):
                audio = pydub.AudioSegment.from_wav(uploaded_file)
                file_type = 'wav'
            elif uploaded_file.name.endswith('mp3'):
                audio = pydub.AudioSegment.from_mp3(uploaded_file)
                file_type = 'mp3'

            save_path = Path(curr_dir) / uploaded_file.name
            save_paths.append(save_path)
            audio.export(save_path, format=file_type)
    return save_paths

def display_wavfile(wavpath):
    audio_bytes = open(wavpath, 'rb').read()
    file_type = Path(wavpath).suffix
    st.audio(audio_bytes, format=f'audio/{file_type}', start_time=0)

#model = create_model()
#model.load_weights("cp.ckpt")

files = upload_and_save_wavfiles()

for wavpath in files:
    display_wavfile(wavpath)
    st.text(files[0])
    st.write(os.getcwd())
    
for i in os.listdir():
    if i.endswith('wav'):
      st.write(i)
      os.remove(i)

'''if st.button("Classify the Genre"):
    data = tf.data.Dataset.list_files(os.getcwd() + '/*.wav')
    data = data.map(preprocess)
    data = data.cache()
    #data = data.shuffle(buffer_size=1000)
    data = data.batch(2)
    #data = data.prefetch(1)

    for i in range(len(files)):
        input = data.as_numpy_iterator().next()
        out = model.predict(input)
        genre = genres_list[out[0]).index(max(out[0])]
        display_wavfile()'''

