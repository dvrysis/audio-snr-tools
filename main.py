import math
import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage
import scipy.io.wavfile

category = 0

inputData = [['/gtzan-m-22.wav','Music'], ['/gtzan-s-22.wav','Speech'], ['yaafe/lvrysis-o-22.wav','Other']]
print(inputData[0][0])

sampleRate, audioData = scipy.io.wavfile.read(inputData[category][0])

print("Input Samplerate: {}Hz".format(sampleRate))
print("Input Duration: {}s".format(len(audioData)/sampleRate))

plot = False

textureSize = 16
fftSize = 2048
stepSize = 2048
timeWidth = 1.0 / sampleRate
frequencyWidth = sampleRate / fftSize

windowData = [0.5-0.5*math.cos(2*math.pi*i/fftSize) for i in range(0, fftSize)]

audioData = audioData.astype(float) / math.pow(2, 15)
frequencies = [i * frequencyWidth for i in range(0, int(fftSize / 2))]

spec = np.ones((textureSize, int(fftSize / 2)))

for index in range(0, len(audioData)-fftSize, stepSize):

    # print "Time: {}".format(float(index)/sampleRate)
    tempData = audioData[index:index+fftSize] * windowData[0:fftSize]
    fft = np.fft.rfft(tempData * timeWidth)
    fft = np.delete(fft, fftSize/2-1)
    mag = np.absolute(fft)
    # mag = np.maximum(mag, 0.000000000000000000001)
    mag = 80*np.log10(mag) + 400
    mag = np.clip(mag, 0, 255)
    mod = int((index / stepSize) % textureSize)

    spec[mod] = mag.astype(int)

    if mod == textureSize - 1:
        startIndex = index - stepSize * (textureSize - 1)
        endIndex = index
        startTime = startIndex / sampleRate
        endTime = endIndex / sampleRate

        times = [float(i)/sampleRate for i in range(startIndex, endIndex)]

        cv.imwrite("F:/Desktop/PhD/Experiments/{}-{}.png".format(inputData[category][1], int(index/stepSize)), spec)
        # print("Output/{}/{}.png".format(inputData[category][1], int(index/stepSize)))
        if plot:

            #Plot waveform
            plt.figure(1)
            plt.subplot(211)
            plt.ylim((-1, 1))
            plt.plot(audioData[startIndex:index])

            # Plot frequency response
            # plt.subplot(212)
            # plt.ylim((0, 255))
            # plt.plot(frequencies, spec[32])
            # plt.show()

            # Plot spectrogram
            # spec = scipy.ndimage.rotate(spec, 90, reshape=True)
            plt.subplot(212)
            plt.gray()
            plt.imshow(spec)
            plt.show()

            # plt.clf()
            # plt.pause(0.05)
            # plt.show()

        # image = scipy.misc.toimage(spec)
        # scipy.misc.toimage(spec)
        # scipy.misc.imsave('Output.jpg', spec)

    ## Calculate energy in time & spectral domain to ensure parseval's theorem
    # powers = np.power(audioData[index:index+fftSize], 2)
    # sum1 = np.sum(powers*timeWidth)
    # powers = np.power(mag[0:511], 2)
    # sum2 = np.sum(powers*binWidth)
    # print "{} and {}".format(sum1, sum2)



