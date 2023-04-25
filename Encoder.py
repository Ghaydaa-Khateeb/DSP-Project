from playsound import playsound

print("**********************************************************************")
print("----- Welcome to The DSP ----- " + "\n" + "Encoder part")
import re

from pydub import AudioSegment
import numpy as np
import contextlib
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pydub import AudioSegment
from scipy.io.wavfile import read
import soundfile

from pydub.generators import Sine

frequncies = {'a': [100,400, 800, 1600], 'b': [100,400, 800, 2400], 'c': [100,400, 800, 4000], 'd': [100,400, 1200, 1600]
    , 'e': [100,400, 1200, 2400], 'f': [100,400, 1200, 4000], 'g': [100,400, 2000, 1600], 'h': [100,400, 2000, 2400]
    , 'i': [100,400, 2000, 4000], 'j': [100,600, 800, 1600], 'k': [100,600, 800, 2400], 'l': [100,600, 800, 4000]
    , 'm': [100,600, 1200, 1600], 'n': [100,600, 1200, 2400], 'o': [100,600, 1200, 4000], 'p': [100,600, 2000, 1600]
    , 'q': [100,600, 2000, 2400], 'r': [100,600, 2000, 4000], 's': [100,1000, 800, 1600], 't': [100,1000, 800, 2400]
    , 'u': [100,1000, 800, 4000], 'v': [100,1000, 1200, 1600], 'w': [100,1000, 1200, 2400], 'x': [100,1000, 1200, 4000]
    , 'y': [100,1000, 2000, 1600], 'z': [100,1000, 2000, 2400], ' ': [100,1000, 2000, 4000],
    'A': [200,400, 800, 1600], 'B': [200,400, 800, 2400], 'C': [200,400, 800, 4000], 'D': [200,400, 1200, 1600]
    , 'E': [200,400, 1200, 2400], 'F': [200,400, 1200, 4000], 'G': [200,400, 2000, 1600], 'H': [200,400, 2000, 2400]
    , 'I': [200,400, 2000, 4000], 'J': [200,600, 800, 1600], 'K': [200,600, 800, 2400], 'L': [200,600, 800, 4000]
    , 'M': [200,600, 1200, 1600], 'N': [200,600, 1200, 2400], 'O': [200,600, 1200, 4000], 'P': [200,600, 2000, 1600]
    , 'Q': [200,600, 2000, 2400], 'R': [200,600, 2000, 4000], 'S': [200,1000, 800, 1600], 'T': [200,1000, 800, 2400]
    , 'U': [200,1000, 800, 4000], 'V': [200,1000, 1200, 1600], 'W': [200,1000, 1200, 2400], 'X': [200,1000, 1200, 4000]
    , 'Y': [200,1000, 2000, 1600], 'Z': [200,1000, 2000, 2400], ' ': [200,1000, 2000, 4000]

              }
print(len(frequncies))

input_string = input("Please Enter strings(with spaces) to make Waveform sound \t : ")
print("\n" + input_string)

# audio = AudioSegment()
res = []
fs=8000
n=np.arange(0,320)
firstfreq = 0

#for i in range(len(input_string)):
 #   if input_string[i].islower() or input_string[i].isspace():
  #      firstfreq = 100
   #     print(input_string[i] + str(firstfreq))

    #elif input_string[i].isupper():
     #   firstfreq = 200
      #  print(input_string[i] + str(firstfreq))

    #lowchara = input_string[i].lower()
    #print("we find i" + "\t" + str(firstfreq) + str(frequncies[lowchara]))


for i in input_string:
    #A1 = Sine(firstfreq, sample_rate=8000, bit_depth=16, ).to_audio_segment(duration=40)
        y=np.cos(frequncies[i][0]*2*np.pi*n/fs)+np.cos(frequncies[i][1]*2*np.pi*n/fs)+np.cos(frequncies[i][2]*2*np.pi*n/fs)+np.cos(frequncies[i][3]*2*np.pi*n/fs)

        res=np.concatenate([res,y])
soundfile.write("out.wav",res, 8000)
    # A3 = np.cos(frequncies[0][0]*2*np.pi*n/fs)
    # A4 = np.cos(frequncies[0][0]*2*np.pi*n/fs)
    # A2 = Sine(frequncies[lowchara].__getitem__(0), sample_rate=8000, bit_depth=16, ).to_audio_segment(duration=40)
    # A3 = Sine(frequncies[lowchara].__getitem__(1), sample_rate=8000, bit_depth=16, ).to_audio_segment(duration=40)
    # A4 = Sine(frequncies[lowchara].__getitem__(2), sample_rate=8000, bit_depth=16, ).to_audio_segment(duration=40)
    # AudioCharacter =  A2 + A3 + A4
    # res.insert(i, AudioCharacter)
    #playsound("out.wav")
#finalresult = res[0]
#for ind in range(1, len(res)):
 #   finalresult += res[ind]
#playsound(finalresult)
#finalresult.export('outputwave.wav', format="wav")
