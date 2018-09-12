from __future__ import unicode_literals
import youtube_dl


ydl_urls = ['https://www.youtube.com/watch?v=Sg-2fPitqrE', 'https://www.youtube.com/watch?v=wzjWIxXBs_s']


for i in range(len(ydl_urls)):

    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'wav',
            'preferredquality': '192',
        }] ,
    }

    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([ydl_urls[i]])
        
        
import sox

#Creating a transformer
tfm = sox.Transformer()
#Trim the audio between 0 to 150 sec
tfm.trim(0, 150)
#Create an output file
tfm.build('The Alchemist - Paulo Coelho - Full Audiobook-Sg-2fPitqrE.wav', 'The Alchemist Audiobook (Trimmed).wav')
tfm.build('CELESTIAL WHITE NOISE _ Sleep Better, Reduce Stress, Calm Your Mind, Improve Focus _ 10 Hour Ambient-wzjWIxXBs_s.wav', 'White noise (Trimmed) .wav')
#Creating a combiner
cbn = sox.Combiner()
#Pitch shift combined audio upto 3 semitones
cbn.pitch(3.0)
#Convert the output to 8000 Hz
cbn.convert(samplerate = 192000, n_channels = 1)
#Create the output file
cbn.build(['The Alchemist Audiobook (Trimmed).wav', 'White noise (Trimmed) .wav'], 'noisy_output.wav', 'mix')




#Importing all the required packages
from scipy.io import wavfile
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

fs, data = wavfile.read('The Alchemist Audiobook (Trimmed).wav')
#Selecting only 1 channel for further use.
data = data[0:150000,0]
#Number of previous samples based on which the next sample is computed.
N = 100

#Extracting each frame of N samples and respective output sample.
X = []
Y = []

#Normalization

d=1/np.max(np.abs(data))
data=data*d

#Range is len(data) - (N + 1) so as to deal with peculiarity at the last sample.
for i in range(len(data)-(N + 1)):

    X.append(data[i:i+N])
    Y.append(data[i+(N + 1)])

#Separate out training, testing and validation data using train_test_split.
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 42)
X_train, X_valid, Y_train, Y_valid = train_test_split(X_train, Y_train, test_size = 0.10, random_state = 42)

m_train = len(X_train)
m_labels = len(Y_train)
m_test = len(X_test)

m_train_n = np.asarray(X_train,float)
m_labels_n = np.asarray(Y_train,float)

m_test_n = np.asarray(X_test)
m_testlabel_n = np.asarray(Y_test)

print('Number of training examples: m_train = ' + str(m_train)+str(type(m_train_n)))
print('Number of training examples: m_train = ' + str(m_labels)+str(type(m_labels_n)))

print('Number of testing examples: m_test = ' + str(m_test))
print('Number of previous samples considered to predict next one: N = ' + str(N))


print(np.size(m_train_n[1,]))
print(np.size(m_labels_n[1,]))



mse_list=[]
input_list=[]
output_list=[]
mse_list_test=[]

def sigmoid(z):
    s=1/(1+np.exp(-z))
    return s
w=np.random.randn(100,)

for i in range(90000):
    input=np.dot(m_train_n[i,],w.T)
    error=m_labels_n[i,] - input 
    output=sigmoid(error)
    if(i%100==0):
        mse=0.5*((error)**2)
        mse_list.append(mse)
    output_list.append(input)
    dw=0.001*(output*m_train_n[i,])
    w= w+dw

for i in range(1,10000):
    pred = np.dot(w.T, m_test_n[i,])
    error_t = m_testlabel_n[i,] - pred
    if (i % 100 == 0):
        mse_t = 0.5 * ((error_t) ** 2)
        mse_list_test.append(mse_t)

plt.plot(mse_list)
plt.title('Mean Squared error (Training Set)')
plt.xlabel('Samples')
plt.ylabel('MSE')

plt.show()

plt.plot(mse_list_test)
plt.title('Mean Squared error (Test Set)')
plt.xlabel('Samples')
plt.ylabel('MSE')

plt.show()

