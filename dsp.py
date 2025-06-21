import numpy as np
import librosa
import matplotlib.pyplot as plt
from pyloudnorm import Meter

def experiment():
    o, sr = librosa.core.load('LVLib-SMO-1\\other.wav')
    # print(o.shape)
    x = np.random.normal(0.0, 0.1, 50000)
    stft = librosa.core.stft(x, 512, 256)
    stft = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    snr = -0

    mel_n = librosa.feature.melspectrogram(x[:50000], 22050, n_fft=512, hop_length=256, power=2.0, n_mels=56)
    p_n = 10*np.log10(np.mean(mel_n))
    mel_n = mel_n * pow(10, 20/10.)

    mel_o = librosa.feature.melspectrogram(o[:50000], 22050, n_fft=512, hop_length=256, power=2.0, n_mels=56)
    p_o = 10 * np.log10(np.mean(mel_o))
    mel_o = mel_o * pow(10, 20/10.)

    print('Signal PW: {:.2f} | Noise PW: {:.2f}'.format(p_o, p_n))

    mel_o = mel_o * pow(10, (p_n-p_o+snr)/10.)
    p_o = 10 * np.log10(np.mean(mel_o))
    print('Signal PW: {:.2f} | Noise PW: {:.2f}'.format(p_o, p_n))

    mel_c = mel_o + mel_n

    plt.figure(1)
    plt.title("Signal Wave...")
    plt.imshow(np.log10(mel_c))
    plt.show()
    # librosa.display.specshow(stft)
    # mel = librosa.

    # plt.figure(1)
    # plt.title("Signal Wave...")
    # plt.plot(stft)
    # plt.show()
    #
    # print(x.shape)

def make_lib(category, step):

    m, sr = librosa.core.load('\music.wav')
    s, sr = librosa.core.load('\speech.wav')
    o, sr = librosa.core.load('\others.wav')
    w, sr = librosa.core.load('\pump.wav')

    length = 22050/256*step
    print(length)
    length = int(length)
    length = int(length*256)
    print(length)

    meter = Meter(sr)

    for i in range(0, m.shape[0]-length, length):

        if category == 0:
            p_s = 20 * np.log10(np.sqrt(np.mean(np.square(m[i:i + length]))))
            l_s = meter.integrated_loudness(m[i:i + length])
            mel_s = librosa.feature.melspectrogram(m[i:i + length], sr, n_fft=512, hop_length=256, power=2, n_mels=56, fmin=100, fmax=8000)
            if i < 1 * m.shape[0] / 3:
                n = s[i:i + length]
            elif i < 2 * m.shape[0] / 3:
                n = o[i:i + length]
            else:
                n = w[i:i + length]  # np.random.normal(0.0, 0.1, length)
        elif category == 1:
            p_s = 20 * np.log10(np.sqrt(np.mean(np.square(s[i:i + length]))))
            l_s = meter.integrated_loudness(s[i:i + length])
            mel_s = librosa.feature.melspectrogram(s[i:i + length], sr, n_fft=512, hop_length=256, power=2, n_mels=56, fmin=100, fmax=8000)
            if i < 1 * m.shape[0] / 3:
                n = m[i:i + length]
            elif i < 2 * m.shape[0] / 3:
                n = w[i:i + length]  # np.random.normal(0.0, 0.1, length)
            else:
                n = o[i:i + length]
        else:
            p_s = 20 * np.log10(np.sqrt(np.mean(np.square(o[i:i + length]))))
            l_s = meter.integrated_loudness(o[i:i + length])
            mel_s = librosa.feature.melspectrogram(o[i:i + length], sr, n_fft=512, hop_length=256, power=2, n_mels=56, fmin=100, fmax=8000)
            if i < 1 * m.shape[0] / 3:
                n = w[i:i + length]  # np.random.normal(0.0, 0.1, length)
            elif i < 2 * m.shape[0] / 3:
                n = m[i:i + length]
            else:
                n = s[i:i + length]

        mp_s = 10 * np.log10(np.mean(mel_s))

        mel_n = librosa.feature.melspectrogram(n, sr, n_fft=512, hop_length=256, power=2, n_mels=56, fmin=100, fmax=8000)
        p_n = 20 * np.log10(np.sqrt(np.mean(np.square(n))))
        l_n = meter.integrated_loudness(n)
        mp_n = 10 * np.log10(np.mean(mel_n))
        print('\nInitial signal pw {:.1f} ({:.1f}) dBFS / {:.1f} LUFS and noise pw {:.1f} ({:.1f}) dBFS / {:.1f} LUFS | SNR is {:.1f} ({:.1f}) dB / {:.1f} LU'.format(mp_s, p_s, l_s, mp_n, p_n, l_n, mp_s - mp_n, p_s - p_n, l_s - l_n))

        if l_s > -120 and l_n > -120:   n = n * pow(10, (l_s - l_n - np.random.normal(12, 3)) / 20.)
        else:                           n = n * pow(10, (mp_s - mp_n - np.random.normal(9, 3)) / 20.)
        # n = n * pow(10, (mp_s - mp_n - np.random.uniform(9, 15)) / 20.)
        # n = n * pow(10, (mp_s - mp_n - 9) / 20.)

        mel_n = librosa.feature.melspectrogram(n, sr, n_fft=512, hop_length=256, power=2, n_mels=56, fmin=100, fmax=8000)
        p_n = 20 * np.log10(np.sqrt(np.mean(np.square(n))))
        l_n = meter.integrated_loudness(n)
        mp_n = 10 * np.log10(np.mean(mel_n))
        print('Final signal pw {:.1f} ({:.1f}) dBFS / {:.1f} LUFS and noise pw {:.1f} ({:.1f}) dBFS / {:.1f} LUFS | SNR is {:.1f} ({:.1f}) dB / {:.1f} LU'.format(mp_s, p_s, l_s, mp_n, p_n, l_n, mp_s - mp_n, p_s - p_n, l_s - l_n))

        # if i < 1 * m.shape[0] / 3:
            #
            # n = n * pow(10, (p_s - p_n - 18) / 10.)
        # elif i < 2 * m.shape[0] / 3:
            #
            # n = n * pow(10, (p_s - p_n - 15) / 10.)
        # else:
            #
            # n = n * pow(10, (p_s - p_n - 12) / 10.)

        if category == 0:
            m[i:i+length] = m[i:i+length] + n
            if i > m.shape[0] - 2*length:
                print('Writing output file...')
                librosa.output.write_wav('Audio SMO v4\\music.wav', m, sr)
        if category == 1:
            s[i:i + length] = s[i:i + length] + n
            if i > m.shape[0] - 2*length:
                print('Writing output file...')
                librosa.output.write_wav('Audio SMO v4\\speech.wav', s, sr)
        if category == 2:
            o[i:i + length] = o[i:i + length] + n
            if i > m.shape[0] - 2*length:
                print('Writing output file...')
                librosa.output.write_wav('Audio SMO v4\\others.wav', o, sr)

make_lib(category=2, step=2)