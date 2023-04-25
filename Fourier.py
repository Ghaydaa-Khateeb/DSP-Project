import soundfile as sf
from scipy.io.wavfile import read
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.fftpack import fft
import numpy as np
from scipy.io import wavfile
import wave
import contextlib
# read audio samples
from scipy.signal import find_peaks

input_data = read("string_5.wav")
audio = input_data[1]
t1 = 0  #Works in milliseconds
t2 = 40
audio1 = []
string =""
with contextlib.closing(wave.open("string_5.wav",'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    print(duration)
print(len('input_data'))
for i in range(int(duration * 1000 / 40)):
    newAudio = AudioSegment.from_wav("string_5.wav")
    newAudio = newAudio[t1: t2]
    newAudio.export('newwave1'+str(i)+'.wav', format = "wav") #Exports to a wav file in the current path.
    t1 += 40
    t2 += 40
    input_data1 = read('newwave1'+str(i)+'.wav')
    audio1 .append(input_data1[1])
for i in audio1:
  f = np.abs(np.fft.fft(i))
  #plt.plot(f)
  freq_steps = np.fft.fftfreq(i.size, d=1/8000 )
  #print(freq_steps)
  y2= len(f) / 2
  y=f[0:int(y2)+10]
  h=2000000
  peaks, cc = find_peaks(y, height=h)
  while len(peaks)<4:
      h=h-100000
      peaks, cc = find_peaks(y, height=h)

  #print(cc)
  print(peaks)
  lm=0
  for k  in peaks:
      peaks[lm]=k*4000/160
      if peaks[lm]>2400 :
          peaks[lm]=4000
      lm=lm+1
  print(peaks)
  m1=0
  m2=0
  m3=0
  m4=0
  if peaks[0]==200or  peaks[1]==200or peaks[2]==200or peaks[3]==200:
      m1=1
  if peaks[0]==600or  peaks[1]==600or peaks[2]==600or peaks[3]==600:
      m2=1
  elif peaks[0]==1000 or  peaks[1]== 1000or peaks[2]==1000or peaks[3]==1000:
      m2=2
  if  peaks[0]==1200or  peaks[1]==1200or peaks[2]==1200or peaks[3]==1200:
      m3=1
  elif peaks[0]==2000or  peaks[1]==2000or peaks[2]==2000or peaks[3]==2000:
      m3=2
  if  peaks[0]==2400or  peaks[1]==2400or peaks[2]==2400or peaks[3]==2400:
      m4=1
  elif peaks[0]==4000or  peaks[1]==4000or peaks[2]==4000or peaks[3]==4000:
      m4=2

  letter='a'
  if m1==1:
      letter='A'

  conv = ord(letter[0])
  conv=conv+(m4+m3*3+m2*9)
  letter=chr(conv)
  if  (m4+m3*3+m2*9)==26:
      letter=' '
  string=string+letter
  print(letter)
  # plot the first 1024 samples
  #y3=4000/y2
  plt.plot( np.abs(freq_steps),f)
  #plt.plot(audio1[0:])
  # label the axes
  plt.ylabel("Amplitude")
  plt.xlabel("Frequency")
  # set the title
  plt.title("Wav")
  # display the plot
  #plt.show()
print(string)