import numpy as np
from scipy.io import wavfile
import sklearn.preprocessing
import matplotlib.pyplot as plt
import math
import tensorflow
import scipy.io
import keras

f = open('/Users/ishaan/Desktop/lab3/phonelist.txt', 'r')
x = f.readlines()
f.close()

X=list(map(lambda x:x.strip(),x))
print(X)

f = open('/Users/ishaan/Desktop/lab3/all_alignments.txt', 'r')
all_phonemes = f.readlines()
f.close()

phonemes=[]
for i in all_phonemes:
    if i.__contains__("AJJacobs"):
        phonemes.append(i)
print(np.shape(phonemes))

curr_list=phonemes[0].split(' ')
print(curr_list[0])
filename=curr_list[0]
curr_list.pop(0)
new_curr_list=[]
for i in curr_list:
    sep = '_'
    new_curr_list.append(i.split(sep, 1)[0])

labels_list=[]
for i in new_curr_list:
    for j in range(len(X)):

        if(i==(X[j])):
            labels_list.append(j)

print(labels_list)

fs1, data1 = wavfile.read('/Users/ishaan/Desktop/lab3/wav/'+ filename +'.wav')
print(fs1)
print(np.size(data1))
data=data1

data_window = []
for i in range(0, (len(data) - 400+1), 160):
    data_window.append(data[i:i + 400])

hamming_window = np.hamming(400)
hammed_data = [hamming_window*segment for segment in data_window]
print("hammed_matrix size", np.shape(hammed_data))
data_rfft = np.fft.rfft(hammed_data)

sample_rate = 16000
min = 0
filters = 40
freq_list = []
floor_list = []

max = (2595 * math.log10(1 + (float(sample_rate/2) / 700)))
print("min = ",min, "max = ", max)
mel_list = np.linspace(min, max, filters + 2)

for mel in mel_list:
    freq = (700 * (10**(mel/ 2595) - 1))
    freq_list.append(freq)

for i in freq_list:
    f = math.floor((400+1)*i/sample_rate)
    floor_list.append(f)
floor1 = floor_list

f2m = np.zeros((40, int(np.floor((400/2) + 1))))
for i in range(1, filters + 1):
    f1, mid, f2 = map(int, floor1[i-1:i+2])
    for k in range(f1, mid):
        f2m[i - 1, k] = (k - floor1[i - 1]) / (floor1[i] - floor1[i - 1])
    for k in range(mid, f2):
        f2m[i - 1, k] = (floor1[i + 1] - k) / (floor1[i + 1] - floor1[i])
print (np.shape(f2m))
print (f2m)

datatotrain=np.dot(np.abs(data_rfft),f2m.T)
print (np.shape(datatotrain))
print(np.count_nonzero((datatotrain)))
print(np.amax((datatotrain)))
xtrain=datatotrain/np.amax(datatotrain)
print(np.amin(datatotrain))
y_train=np.asarray(labels_list)

from keras.layers import Input, Dense
from keras.models import Model
from keras import optimizers
from keras.callbacks import TensorBoard
from time import time
from keras import Sequential
from keras.utils import np_utils
labels_train = np_utils.to_categorical(y_train,46)
model = Sequential()
model.add(Dense(128, input_dim=40, activation='relu'))
model.add(Dense(46, activation='relu'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(xtrain,labels_train , epochs=150, batch_size=10,verbose=1)