import subprocess
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import math
import scipy.signal
import youtube_dl
import sox
import scipy.io.wavfile
import sklearn.preprocessing
import tensorflow
import scipy.io
import keras

from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers


def audio_download(url, audio_name):

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }],
    'outtmpl': '%(id)s.%(ext)s',
    }
    
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        op_result = ydl.extract_info(url, download= True)
        
    audio_file = op_result['id']+'.wav'
    
    #convert to mono channel and 16000Hz sampling rate
    new1 = sox.Transformer()
    new1.channels(1)
    new1.rate(16000)
    new_audio = audio_name+audio_file    
    new1.build(audio_file, new_audio) 
    
    return new_audio


audioS = audio_download('https://www.youtube.com/watch?v=kcW4ABcY3zI&t=5s', 'sound_')
freqS, S = scipy.io.wavfile.read(audioS)
audioN = audio_download('https://www.youtube.com/watch?v=ElPpSvehdTI', 'noise_')
freqN, N = scipy.io.wavfile.read(audioN)

speech_train = S[25000:290000]
noise_train = N[25000:290000]

speech_test = S[290001:342000]
noise_test = N[290001:342000]


def norm_d(input_values):
    max_val = np.max(np.abs(input_values))
    input_values = np.divide(input_values,max_val)
    return input_values

def hamm_fn(ip_mtx):
    hamm_win = np.hamming(512)
    hamm_mtx = [hamm_win*segment for segment in ip_mtx]
    #print("hammed_matrix size", np.shape(hammed_matrix))
    return hamm_mtx

def gen_dataset(audio_samples, w_size, step_size):
    x = []
    for i in range(0, (len(audio_samples) - w_size+1), step_size):
        x.append(audio_samples[i:i + w_size])
    return x

sampling_rate = 16000

#print(noise_train.shape)

w = 0.2
mixed_train = speech_train+noise_train*w
scipy.io.wavfile.write("op_mix_train.wav", sampling_rate, mixed_train.astype('int16'))


mixed_train_windowed = gen_dataset(mixed_train, 512, 128)
mixed_train_hammed = hamm_fn(mixed_train_windowed)
mixed_train_rfft = np.fft.rfft(mixed_train_hammed)

speech_train_windowed = gen_dataset(speech_train, 512, 128)
speech_train_hammed = hamm_fn(speech_train_windowed)
speech_train_rfft = np.fft.rfft(speech_train_hammed)

w = 0.8
mixed_test = speech_test+noise_test*w
scipy.io.wavfile.write("op_mix_test.wav", sampling_rate, mixed_test.astype('int16'))
mixed_test_windowed = gen_dataset(mixed_test, 512, 128)
mixed_test_hammed = hamm_fn(mixed_test_windowed)
mixed_test_rfft = np.fft.rfft(mixed_test_hammed)

speech_test_windowed = gen_dataset(speech_test, 512, 128)
speech_test_hammed = hamm_fn(speech_test_windowed)
speech_test_rfft = np.fft.rfft(speech_test_hammed)

# Creating the mel filter bank
sampling_rate = 16000
m1 = 0
m2 = (2595 * math.log10(1 + (float(sampling_rate/2) / 700))) 
filters = 40
mel_step = (m2 - m1)/(filters+1)
mel_list = np.linspace(m1, m2, filters + 2)

freq_list = []
for mel in mel_list:
    freq = (700 * (10**(mel/ 2595) - 1))
    freq_list.append(freq)

floored_freq_list = []
for i in freq_list:
    f = math.floor((512+1)*i/16000)
    floored_freq_list.append(f)

bin_index = floored_freq_list

# the line generation for the bins
freq2mel = np.zeros((40, int(np.floor((512/2) + 1))))
for i in range(1, filters + 1):
    f1, mid, f2 = map(int, bin_index[i-1:i+2])
    for k in range(f1, mid):
        freq2mel[i - 1, k] = (k - bin_index[i - 1]) / (bin_index[i] - bin_index[i - 1])
    for k in range(mid, f2):
        freq2mel[i - 1, k] = (bin_index[i + 1] - k) / (bin_index[i + 1] - bin_index[i])

# Calculate mel values 

mixed_train_mel=np.dot(np.abs(mixed_train_rfft),freq2mel.T)
speech_train_mel=np.dot(np.abs(speech_train_rfft),freq2mel.T)
mixed_test_mel=np.dot(np.abs(mixed_test_rfft),freq2mel.T)
speech_test_mel=np.dot(np.abs(speech_test_rfft),freq2mel.T)

#normalization and including the phase info by including mel2freq conversion
mixed_train_mel = norm_d(mixed_train_mel)
speech_train_mel = norm_d(speech_train_mel)
mixed_test_mel = norm_d(mixed_test_mel)
speech_test_mel = norm_d(speech_test_mel)

xtrain = mixed_train_mel
ytrain = np.divide(speech_train_mel, mixed_train_mel)

xtest = mixed_test_mel
ytest = np.divide(speech_test_mel, mixed_test_mel)

if np.count_nonzero(np.isnan(ytrain)) > 0 :
    rows, cols = np.shape(ytrain)
    for i in range(rows):
        for j in range(cols):
            if np.isnan(ytrain[i][j]):
                ytrain[i][j] = 0

#print(np.count_nonzero(np.isnan(ytrain)))

# Training the auto encoder

input_audio=Input(shape=(40,))
encoded_1 = Dense(100, activation='tanh')(input_audio)
encoded_2 = Dense(300, activation='tanh')(encoded_1)
decoded_1 = Dense(300, activation='tanh')(encoded_2)
decoded_2 = Dense(40, activation='sigmoid')(decoded_1)

#using Adam optimizer
ADAM = optimizers.Adam(lr=0.01)
autoencoder = Model(input_audio, decoded_2)
autoencoder.compile(optimizer=ADAM, loss='mse', metrics=['accuracy'])

autoencoder.summary()
csv_logger = keras.callbacks.CSVLogger('training.log')
history = autoencoder.fit(xtrain, ytrain, epochs = 30, validation_split=0.2, shuffle=True, callbacks=[csv_logger])
metrics=autoencoder.evaluate(xtest, ytest, verbose=1)


print(history.history.keys())
print(metrics)

predicted_output=autoencoder.predict(xtest)
predicted_mel_output = np.dot(predicted_output, freq2mel)
predicted_freq = predicted_mel_output * speech_test_rfft
predicted_irfft = np.fft.irfft(predicted_freq)

#rebuilding the audio signal
rebuilt_window_signal = np.zeros(54000)
for i in range(np.shape(predicted_irfft)[0]):
    for j in range(512):
        rebuilt_window_signal[(i*128)+j] += predicted_irfft[i,j]

rebuilt_normalized=rebuilt_window_signal/max(rebuilt_window_signal)*32767
scipy.io.wavfile.write("op_rebuilt.wav", sampling_rate, rebuilt_normalized.astype('int16'))

def plot_acc_loss(history):
    #  "Accuracy"
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('accuracy of the model')
    plt.ylabel('Accuracy')
    plt.xlabel('Epochs')
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()
    # "Loss"
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.show()

plot_acc_loss(history)


