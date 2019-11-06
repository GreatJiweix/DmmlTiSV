import torch.utils.data as data

from PIL import Image

import os
import os.path
import numpy as np
import sys
import audio_processing
import glob
#import vad
from scipy.io import wavfile
import librosa
from scipy.signal.windows import hamming
import soundfile as sf
from python_speech_features import fbank, delta
import constants as c
from torch.utils.data import DataLoader, Dataset

SAMPLE_RATE = 16000
FEATURE = 'fft'#
FEATURE_LEN = 161#
WIN_LEN = 0.02#
WIN_STEP = 0.01#

N_FFT = int(WIN_LEN * SAMPLE_RATE)#
HOP_LEN = int(WIN_STEP * SAMPLE_RATE)#

N_FRAMES = 300#
DURATION = (N_FRAMES - 1) * WIN_STEP#
N_SAMPLES = int(DURATION * SAMPLE_RATE)#

N_TEST_FRAMES = 300#
TEST_DURATION = (N_TEST_FRAMES - 1) * WIN_STEP#
N_TEST_SAMPLES = int(TEST_DURATION * SAMPLE_RATE)#





def wav_loader_stfft(path):

    def load_audio(filename, start=0, stop=None, resample=True):
        sr = SAMPLE_RATE
        y, sr = sf.read(filename, start=start, stop=stop, dtype='float32', always_2d=True)
        y = np.squeeze(y)
        return y, sr
    #print(path)
    y, sr = load_audio(path)
    num_frame = 300

    S = librosa.core.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, window=hamming)#
    feature, _ = librosa.magphase(S)
    npy = np.log1p(feature)#
    npy = npy.transpose()

    l = npy.shape[0]
    if l <= num_frame:
        new = np.zeros((num_frame, 161))
        new[:l, :] = npy
        new[num_frame-l:, :] = npy[:l, :]
        npy = new
    npy = (npy - np.mean(npy, axis=0)) / (np.std(npy, axis=0) + 2e-12)

    return npy

def normalize_frames(m,Scale=True):
    if Scale:
        return (m - np.mean(m, axis=0)) / (np.std(m, axis=0) + 2e-12)
    else:
        return (m - np.mean(m, axis=0))


def mk_MFB(filename, sample_rate=c.SAMPLE_RATE,use_delta = c.USE_DELTA,use_scale = c.USE_SCALE,use_logscale = c.USE_LOGSCALE):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    #audio = audio.flatten()


    filter_banks, energies = fbank(audio, samplerate=sample_rate, nfilt=c.FILTER_BANK, winlen=0.025)

    if use_logscale:
        filter_banks = 20 * np.log10(np.maximum(filter_banks,1e-5))

    if use_delta:
        delta_1 = delta(filter_banks, N=1)
        delta_2 = delta(delta_1, N=1)

        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        delta_1 = normalize_frames(delta_1, Scale=use_scale)
        delta_2 = normalize_frames(delta_2, Scale=use_scale)

        frames_features = np.hstack([filter_banks, delta_1, delta_2])
    else:
        filter_banks = normalize_frames(filter_banks, Scale=use_scale)
        frames_features = filter_banks 

    #print(frames_features.shape)



    #np.save(filename.replace('.wav', '.npy'),frames_features)

    return frames_features



root_dir = 
out_dir = 

dirs = os.listdir(root_dir)
dirs.sort()




class my_dataset(Dataset):
    def __init__(self, list_data):
        self.list_data = list_data
    def __getitem__(self, item):
        img = self.list_data[item]
        im = mk_MFB(img)
        #print(im.shape)
        img_out = img.replace(root_dir, out_dir)
        img_out = img_out.replace(".wav", ".npy")
        img_out = img_out.replace(".WAV", ".npy")
        img_jpg = img_out.replace(".npy", ".jpg")
        #print(img, img_out, img_jpg)
        
        np.save(img_out, im)

        try:
            os.remove(img_jpg)
        except:
            pass

        return img_out
    def __len__(self):
        return len(self.list_data)

def get_lists(root_dir):
    dirs = os.listdir(root_dir)
    dirs.sort()
    imgs = []

    for dir in dirs:
        print(dir)
        dir = os.path.join(root_dir, dir)    	
        for img in glob.glob(os.path.join(dir, "*.wav")):
            imgs.append(img)
        for img in glob.glob(os.path.join(dir, "*.WAV")):
            imgs.append(img)

        imgs.sort()

    return imgs
list_data = get_lists(root_dir)
print(len(list_data))
dataset = my_dataset(list_data)
loader = DataLoader(dataset, batch_size=16, num_workers=8, drop_last=True)

for i, imgs in enumerate(loader):
	print(imgs)


