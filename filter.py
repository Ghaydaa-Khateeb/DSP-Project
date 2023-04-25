from scipy import signal

import scipy
from scipy.signal import butter, lfilter, sosfilt, freqz
import scipy.io.wavfile
import scipy.signal
import numpy as np
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

import matplotlib.pyplot as plt
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

scipy.signal.filtfilt
input_data = read("capital.wav")
audio = input_data[1]
t1 = 0  # Works in milliseconds
t2 = 40
audio1 = []
string = ""
with contextlib.closing(wave.open("capital.wav", 'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    duration = frames / float(rate)
    print(duration)
print(len('input_data'))
for i in range(int(duration * 1000 / 40)):
    newAudio = AudioSegment.from_wav("capital.wav")
    newAudio = newAudio[t1: t2]
    newAudio.export('newwave' + str(i) + '.wav', format="wav")  # Exports to a wav file in the current path.
    t1 += 40
    t2 += 40
    input_data1 = read('newwave' + str(i) + '.wav')
    audio1.append(input_data1[1])
for i in audio1:
    f = np.abs(np.fft.fft(i))
    # plt.plot(f)
    freq_steps = np.fft.fftfreq(i.size, d=1 / 8000)

    plt.figure(1)
    plt.clf()
    letterfrq = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for order in [9]:
        letterfrq[0] = b, a = butter_bandpass(60, 140, 300, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((300 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # ------100
        letterfrq[1] = b, a = butter_bandpass(160, 240, 500, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((500 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -------200
        letterfrq[2] = b, a = butter_bandpass(360, 440, 1000, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((1000 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----400
        letterfrq[3] = b, a = butter_bandpass(760, 840, 1950, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((1950 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----800
        letterfrq[4] = b, a = butter_bandpass(1560, 1640, 5500, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((5500 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----1600
        letterfrq[5] = b, a = butter_bandpass(560, 640, 1500, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((1500 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----600
        letterfrq[6] = b, a = butter_bandpass(1160, 1240, 3200, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((3200 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----1200
        letterfrq[7] = b, a = butter_bandpass(2360, 2440, 9000, order=order)
        w, h = freqz(b, a, worN=2000)

        # = scipy.signal.filtfilt(b, a, f)
        #plt.plot((9000 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)

        # -----2400
        letterfrq[8] = b, a = butter_bandpass(960, 1040, 2600, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((2600 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----1000
        letterfrq[9] = b, a = butter_bandpass(1960, 2040, 6000, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((6000 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----2000
        letterfrq[10] = b, a = butter_bandpass(3920, 4080, 15300, order=order)
        w, h = freqz(b, a, worN=2000)
        #plt.plot((15300 * 0.5 / np.pi) * w, abs(h), label="order = %d" % order)
        # -----4000
        let=[0,0,0,4000]
        x=0
        y23 = len(f) / 2
        y = f[0:int(y23) + 10]
        hh = 2000000
        peaks, cc = find_peaks(y, height=hh)
        while len(peaks) < 4:
            hh = hh - 100000
            peaks, cc = find_peaks(y, height=hh)
        hh=1500000
        lm = 0
        for k in peaks:
            peaks[lm] = k * 4000 / 160
            lm = lm + 1
        print(peaks)
        y1 = np.abs(np.fft.fft(butter_bandpass_filter(i, 50, 250, 16000, order=5)))
        y22 = len(y1) / 2
        y1 = y1[0:int(y22) + 10]
        peaks, cc = find_peaks(y1, height=hh)
        peaks = peaks * 4000 / 160
        if len(peaks)>0:
            let[x]=peaks[0]
            x=x+1
            b, a = butter_bandpass(60, 140, 300, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((300 * 0.5 / np.pi) * w, abs(h)*3500000, label="order = %d" % order)
        print(peaks)
        y2 = np.abs(np.fft.fft(butter_bandpass_filter(i, 300, 500, 16000, order=5)))
        y22 = len(y2) / 2
        y2 = y2[0:int(y22) + 10]
        peaks, cc = find_peaks(y2, height=hh)
        peaks = peaks * 4000 / 160
        if  len(peaks)  >0:
            let[x]=peaks[0]
            x=x+1
            b, a = butter_bandpass(160, 240, 500, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((500 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        print(peaks)
        y3 = np.abs(np.fft.fft(butter_bandpass_filter(i, 700, 900, 16000, order=5)))
        y22 = len(y3) / 2
        y3 = y3[0:int(y22) + 10]
        peaks, cc = find_peaks(y3, height=hh)
        peaks = peaks * 4000 / 160
        print("ppppppp")
        print(peaks)
        print(hh)
        if len(peaks) > 0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(360, 440, 1000,order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((1000 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y4 = np.abs(np.fft.fft(butter_bandpass_filter(i, 1100, 1300, 16000, order=5)))
        y22 = len(y4) / 2
        y4 = y4[0:int(y22) + 10]
        peaks, cc = find_peaks(y4, height=hh)
        peaks = peaks * 4000 / 160
        print(peaks)
        if len(peaks) > 0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(560, 640, 1500, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((1500 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y5 = np.abs(np.fft.fft(butter_bandpass_filter(i, 1900, 2100, 16000, order=5)))
        y22 = len(y5) / 2
        y5 = y5[0:int(y22) + 10]
        peaks, cc = find_peaks(y5, height=hh)
        peaks = peaks * 4000 / 160
        print(peaks)
        if  len(peaks)  >0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(960, 1040, 2600, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((2600 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y6 = np.abs(np.fft.fft(butter_bandpass_filter(i, 1500, 1700, 16000, order=5)))
        y22 = len(y6) / 2
        y6 = y6[0:int(y22) + 10]
        peaks, cc = find_peaks(y6, height=hh)
        peaks = peaks * 4000 / 160
        print(peaks)
        if  len(peaks)  >0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(760, 840, 1950, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((1950 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y7 = np.abs(np.fft.fft(butter_bandpass_filter(i, 2300, 2500, 16000, order=5)))
        y22 = len(y7) / 2
        y7 = y7[0:int(y22) + 10]
        peaks, cc = find_peaks(y7, height=hh)
        peaks = peaks * 4000 / 160
        print(peaks)
        if  len(peaks)  >0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(1160, 1240, 3200, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((3200 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y8 = np.abs(np.fft.fft(butter_bandpass_filter(i, 3900, 4100, 16000, order=5)))
        y22 = len(y8) / 2
        y8 = y8[0:int(y22) + 10]
        peaks, cc = find_peaks(y8, height=hh)
        peaks = peaks * 4000 / 160
        print(peaks)
        if  len(peaks)  >0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(1960, 2040, 6000, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((6000 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y9 = np.abs(np.fft.fft(butter_bandpass_filter(i, 3100, 3300, 16000, order=5)))
        y22 = len(y9) / 2
        y9 = y9[0:int(y22) + 10]
        peaks, cc = find_peaks(y9, height=hh)
        peaks = peaks * 4000 / 160
        print(peaks)
        if  len(peaks)  >0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(1560, 1640, 5500, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((5500* 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y10 = np.abs(np.fft.fft(butter_bandpass_filter(i, 4700, 4900, 16000, order=5)))
        y22 = len(y10) / 2
        y10 = y10[0:int(y22) + 10]
        peaks, cc = find_peaks(y10, height=hh)
        peaks = peaks * 4000 / 160
        print(peaks)
        if  len(peaks)  >0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(2360, 2440, 9000, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((9000 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        y11 = np.abs(np.fft.fft(butter_bandpass_filter(i, 7900, 8100, 18000, order=5)))

        y22 = len(y11) / 2
        y11 = y11[0:int(y22) + 10]
        peaks, cc = find_peaks(y11, height=hh)
        peaks=peaks*4000/160
        print(peaks)
        if  len(peaks)  >0:
            let[x] = peaks[0]
            x = x + 1
            b, a = butter_bandpass(3920, 4080, 15300, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((15300 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)
        print(let)
        if  let[3]==4000:
            b, a = butter_bandpass(3920, 4080, 15300, order=order)
            w, h = freqz(b, a, worN=2000)
            plt.plot((15300 * 0.5 / np.pi) * w, abs(h) * 3500000, label="order = %d" % order)

        plt.plot(np.abs(freq_steps),f)
        peaks=let
        m1 = 0
        m2 = 0
        m3 = 0
        m4 = 0
        if peaks[0] == 200 or peaks[1] == 200 or peaks[2] == 200 or peaks[3] == 200:
            m1 = 1
        if peaks[0] == 600 or peaks[1] == 600 or peaks[2] == 600 or peaks[3] == 600:
            m2 = 1
        elif peaks[0] == 1000 or peaks[1] == 1000 or peaks[2] == 1000 or peaks[3] == 1000:
            m2 = 2
        if peaks[0] == 1200 or peaks[1] == 1200 or peaks[2] == 1200 or peaks[3] == 1200:
            m3 = 1
        elif peaks[0] == 2000 or peaks[1] == 2000 or peaks[2] == 2000 or peaks[3] == 2000:
            m3 = 2
        if peaks[0] == 2400 or peaks[1] == 2400 or peaks[2] == 2400 or peaks[3] == 2400:
            m4 = 1
        elif peaks[0] == 4000 or peaks[1] == 4000 or peaks[2] == 4000 or peaks[3] == 4000:
            m4 = 2

        letter = 'a'
        if m1 == 1:
            letter = 'A'

        conv = ord(letter[0])
        conv = conv + (m4 + m3 * 3 + m2 * 9)
        letter = chr(conv)
        if (m4 + m3 * 3 + m2 * 9) == 26:
            letter = ' '
        string = string + letter
        print(letter)


        #plt.show()
print(string)
