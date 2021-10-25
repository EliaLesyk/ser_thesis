import numpy as np
import pylab
import skimage.measure
import wave
import python_speech_features as psf
import librosa
import math
import matplotlib.pyplot as plt
import scipy.io.wavfile as wav


SPECTR_DUR = 128

NUMCEP = 40
MFCC_DUR = 50
MFCC_STEP = 25


def generate_spectrogram(wav_file, view=False):
    MAX_SPETROGRAM_LENGTH = 999  # 8 sec
    MAX_SPETROGRAM_TIME_LENGTH_POOLED = SPECTR_DUR
    MAX_SPETROGRAM_FREQ_LENGTH_POOLED = SPECTR_DUR

    def get_wav_info(wav_file):
        wav = wave.open(wav_file, 'r')
        frames = wav.readframes(-1)
        #sound_info = pylab.fromstring(frames, 'Int16')
        sound_info = pylab.fromstring(frames, pylab.uint8)
        frame_rate = wav.getframerate()
        wav.close()
        return sound_info, frame_rate

    """Based on https://dzone.com/articles/generating-audio-spectrograms"""

    """Loading wav file"""
    sound_info, frame_rate = get_wav_info(wav_file)

    """Creating spectrogram"""
    spec, freqs, times, axes = pylab.specgram(sound_info, Fs=frame_rate)

    """Checking dimensions of spectrogram"""
    assert spec.shape[0] == freqs.shape[0] and spec.shape[1] == times.shape[0], "Original dimensions of spectrogram are inconsistent"

    """Extracting a const length spectrogram"""
    times = times[:MAX_SPETROGRAM_LENGTH]
    spec = spec[:, :MAX_SPETROGRAM_LENGTH]
    assert spec.shape[1] == times.shape[0], "Dimensions of spectrogram are inconsistent after change"

    spec_log = np.log(spec)
    spec_pooled = skimage.measure.block_reduce(spec_log, (1, 8), np.mean)
    spec_cropped = spec_pooled[:MAX_SPETROGRAM_FREQ_LENGTH_POOLED, :MAX_SPETROGRAM_TIME_LENGTH_POOLED]
    spectrogram = np.zeros((MAX_SPETROGRAM_FREQ_LENGTH_POOLED, MAX_SPETROGRAM_TIME_LENGTH_POOLED))
    spectrogram[:, :spec_cropped.shape[1]] = spec_cropped

    if view:
        plt.imshow(spec_cropped, cmap='hot', interpolation='nearest')
        plt.show()
    #print(spectrogram.shape)
    return spectrogram


def generate_mspectr(file_audio, view=False):
    y, sr = librosa.load(file_audio, sr=16000)      # works with errors without sr=16k
    mspectr = librosa.feature.melspectrogram(y, sr, n_fft=4096, hop_length=int(0.01*sr), win_length=int(0.025*sr), \
                                             window='hann')
    #print("mspectr.shape", mspectr.shape)      # (128, 120-1100)
    if view:
        plt.imshow(mspectr, cmap='hot', interpolation='nearest')
        plt.show()
    return mspectr


def generate_mfcc(file_audio, view=False):
    fs, sig = wav.read(file_audio)  # fs is sample rate
    mfcc = psf.mfcc(sig, fs, numcep=NUMCEP, nfilt=NUMCEP) # numcep=128 -> (dur, numcep), should be flipped
    #sig, fs = librosa.load(file_audio)
    #mfcc = librosa.feature.mfcc(sig, fs, n_mfcc=NUMCEP, n_fft=2048)

    # np.flipud - Reverse the order of elements along axis 0 (up/down) -> from (dur, numcep) to (numcep, dur)
    mfcc_reord = np.flipud(mfcc).T
    if view:
        plt.figure(figsize=(20,5))
        plt.title('MFCC')
        plt.imshow(mfcc_reord, aspect='auto', extent=[0, len(sig)/fs, 0,  10])
        plt.ylabel('Coefficients', fontsize=18)
        plt.xlabel('Time [sec]', fontsize=18)
        plt.tight_layout()
        plt.show()
    #print("mfcc shape", mfcc_reord.shape)
    return mfcc_reord


'''
https://github.com/sleekEagle/audio_processing/blob/master/mix_noise.py
Signal to noise ratio (SNR) can be defined as 
SNR = 20*log(RMS_signal/RMS_noise)
where RMS_signal is the RMS value of signal and RMS_noise is that of noise.
      log is the logarithm of 10
*****additive white gausian noise (AWGN)****
 - This kind of noise can be added (arithmatic element-wise addition) to the signal
 - mean value is zero (randomly sampled from a gausian distribution with mean value of zero. standard daviation can varry)
 - contains all the frequency components in an equal manner (hence "white" noise) 
'''


def add_white_noise(file_audio, SNR=10):
    '''
    given a signal and desired SNR (in dB), this gives the required AWGN
    what should be added to the signal to get the desired SNR
    '''
    #signal, sr = librosa.load(file_audio, sr=16000)
    sr, signal = wav.read(file_audio)
    signal = np.interp(signal, (signal.min(), signal.max()), (-1, 1))
    #RMS value of signal
    RMS_s = math.sqrt(np.mean(signal**2))
    #RMS values of noise
    RMS_n = math.sqrt(RMS_s**2/(pow(10, SNR/10)))
    #Additive white gausian noise. Thereore mean=0
    #Because sample length is large (typically > 40000)
    #we can use the population formula for standard daviation.
    #because mean=0 STD=RMS
    STD_n = RMS_n
    noise = np.random.normal(0, STD_n, signal.shape[0])
    signal_with_noise = signal + noise
    # save file to the folder
    file_audio_name = file_audio.split('.wav')[0]
    file_final_name = "{}.wav".format(file_audio_name + "_with_{}dB_noise".format(SNR))
    return wav.write(file_final_name, sr, signal_with_noise)


"""
from scipy.fftpack import dct #Discrete Cosine Transform
from scipy.io.wavfile import read # libreria para lectura de archivos de audio
#from IPython.display import Audio # para escuchar la senal
import numpy as np
#import matplotlib.pyplot as plt # libreria usada para graficas
from sklearn import preprocessing

def extract_windows(signal, size, step):
    # make sure we have a mono signal
    assert (signal.ndim == 1)

    n_frames = int((len(signal) - size) / step)

    # extract frames
    windows = [signal[i * step: i * step + size]
               for i in range(n_frames)]

    # stack (each row is a window)
    return np.vstack(windows)


def powerspec(X, rate, size, n_padded_min=0):
    # hanning window
    X *= np.hanning(size)

    # zero padding to next power of 2
    if n_padded_min == 0:
        n_padded = max(n_padded_min, int(2 ** np.ceil(np.log(size) / np.log(2))))
    else:
        n_padded = n_padded_min

    # Fourier transform
    Y = np.fft.fft(X, n=n_padded)
    Y = np.absolute(Y)

    # non-redundant part
    m = int(n_padded / 2) + 1
    Y = Y[:, :m]

    return np.abs(Y) ** 2, n_padded


def lifter(cepstra, L=22):
    if L > 0:
        nframes, ncoeff = np.shape(cepstra)
        n = np.arange(ncoeff)
        lift = 1 + (L / 2) * np.sin(np.pi * n / L)
        return lift * cepstra
    else:   # values of L <= 0, do nothing
        return cepstra


def generate_mfcc(file_audio, view=False):
    fs, audio_input = read(file_audio)  # fs is sample rate
    #t = np.arange(0, float(len(audio_input)) / fs, 1.0 / fs)  # Vector de tiempo

    # Standardizing
    scaler = preprocessing.StandardScaler()  # z = (x - mean) / std_dev
    standard_X = np.hstack(scaler.fit_transform(np.vstack(audio_input)))

    SIZE = int(0.020 * fs)  # 20 miliseconds window -> to samples 0.020*fs
    STEP = int(0.010 * fs)  # 10 miliseconds hop
    ALPHA = 0.97  # This value is standard according to the State-of-the-art

    SAMPLE_RATE = 8000  # According to the Frequencies of the human speech
    LOW_FREQ = 0
    HIGH_FREQ = SAMPLE_RATE / 2
    N_FILT = 128  # Number of triangular filters

    emphasized_signal = np.append(standard_X[0], standard_X[1:] - ALPHA * standard_X[:-1])

    # Windows of 20 ms with hop/step of 10 ms
    frames = extract_windows(emphasized_signal, SIZE, STEP)
    frames *= np.hamming(SIZE)  # size=20ms

    pow_frames, nfft = powerspec(np.vstack(frames), fs, SIZE, 1024)
    # To improve the visualization we use the log-scale
    #spect = np.flipud(np.log(pow_frames).T)

    # compute points evenly spaced in mels
    lowmel = 2595 * np.log10(1 + LOW_FREQ / 700.0)  # convert Hrz into Mels
    highmel = 2595 * np.log10(1 + HIGH_FREQ / 700.0)
    #print(lowmel, highmel)
    melpoints = np.linspace(lowmel, highmel, N_FILT + 2)
    # our points are in Hz, but we use fft bins, so we have to convert
    #  from Hz to fft bin number
    bin = np.floor((nfft + 1) * (700 * (10 ** (melpoints / 2595.0) - 1)) / SAMPLE_RATE)

    fbank = np.zeros([N_FILT, int(nfft / 2 + 1)])
    for j in range(0, N_FILT):
        for i in range(int(bin[j]), int(bin[j + 1])):
            fbank[j, i] = (i - bin[j]) / (bin[j + 1] - bin[j])
        for i in range(int(bin[j + 1]), int(bin[j + 2])):
            fbank[j, i] = (bin[j + 2] - i) / (bin[j + 2] - bin[j + 1])

    # Multiply Mel filters/ filter banks
    spec_mel = np.dot(pow_frames, fbank.T)
    spec_mel = np.where(spec_mel == 0.0, np.finfo(float).eps, spec_mel)
    spec_mel = np.log(spec_mel)

    NUM_CEPS = 12  # Number of coefficients to extract
    CEP_LIFTER = 22

    mfcc = dct(spec_mel, type=2, axis=1, norm='ortho')[:, :NUM_CEPS]
    mfcc = lifter(mfcc, CEP_LIFTER)
    mfcc_reord = np.flipud(mfcc).T      # reverse from up to down along axis 0

    if view:
        plt.figure(figsize=(20, 5))
        # plt.title('Spectrogram')
        plt.title('MFCC')
        plt.imshow(mfcc_reord, aspect='auto', extent=[0, len(standard_X) / fs, 0, 10])
        plt.ylabel('Coefficients', fontsize=18)
        plt.xlabel('Time [sec]', fontsize=18)
        plt.tight_layout()
        plt.show()

    return mfcc_reord
"""

"""
def read_iemocap(labels_df):
    # creating dataframe
    #columns = ['wav_file', 'label', 'gender', 'sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence', 'harmonic',
             #  'auto_corr_max', 'auto_corr_std']
    #features = pd.DataFrame(columns=columns)
    filt_df = labels_df[labels_df['emotion'].isin(emotions_used)]
    #filt_df['gender'] = filt_df['gender'].map(dict(zip(['M', 'F'], [0, 1])))
    feature_list = filt_df[['File Name', 'emotion', 'gender']].copy()  # wav_file, label
    #print(feature_list)
    for session in sessions:
        for index, row in feature_list[feature_list['File Name'].str.contains('Ses0{}'.format(session[-1]))].iterrows():
        #for index, row in labels_df.iterrows():
            try:
                file_name = row['File Name']
                #gender = row['gender']
                #if row['emotion'] in emotions_used:
                #label = row['emotion']
                file_path = data_path + session + "/" + file_name + ".wav"
                #y = audio_vectors[file_name]
                y, _sr = librosa.load(file_path, sr=44100)

                # (?) check whether y is signal
                # feature_list['signal'] = y

                sig_mean = np.mean(abs(y))
                feature_list['sig_mean'] = sig_mean
                feature_list['sig_std'] = np.std(y)

                rmse = librosa.feature.rms(y + 0.0001)[0] # librosa.feature.rmse isn't supported
                feature_list['rmse_mean'] = np.mean(rmse)
                feature_list['rmse_std'] = np.std(rmse)

                silence = 0
                for e in rmse:
                    if e <= 0.4 * np.mean(rmse):
                        silence += 1
                silence /= float(len(rmse))
                feature_list['silence'] = silence  # silence

                y_harmonic = librosa.effects.hpss(y)[0]
                feature_list['harmonic'] = np.mean(y_harmonic) * 1000  # harmonic (scaled by 1000)

                # based on the pitch detection algorithm mentioned here:
                # http://access.feld.cvut.cz/view.php?cisloclanku=2009060001
                cl = 0.45 * sig_mean
                center_clipped = []
                for s in y:
                    if s >= cl:
                        center_clipped.append(s - cl)
                    elif s <= -cl:
                        center_clipped.append(s + cl)
                    elif np.abs(s) < cl:
                        center_clipped.append(0)
                auto_corrs = librosa.core.autocorrelate(np.array(center_clipped))
                auto_corr_max = 1000 * np.max(auto_corrs)/len(auto_corrs)
                feature_list['auto_corr_max'] = auto_corr_max # scaled by 1000
                feature_list['auto_corr_std'] = np.std(auto_corrs)  # auto_corr_std
                #print(feature_list)
                #features.append(pd.DataFrame(feature_list, index=columns).transpose(), ignore_index=True)
                #print("Features dataset len:", len(features))
            except:
                print('Some exception occured for ', file_name)
    #print("Feature list len:", len(feature_list[2]))
    #return len(feature_list)

    #columns = ['wav_file', 'label', 'gender', 'sig_mean', 'sig_std', 'rmse_mean', 'rmse_std', 'silence',
                       # 'harmonic', 'auto_corr_max', 'auto_corr_std']
    #features = pd.DataFrame(feature_list, index=columns)
    return feature_list
"""
