import torch.utils.data as data

from PIL import Image

import os
import os.path
import numpy as np
import sys
import audio_processing
#import vad
from scipy.io import wavfile
import librosa
from scipy.signal.windows import hamming
import soundfile as sf
import random

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

noise_dir = "/home/xjw/data/RIRS_NOISES/pointsource_noises/"
noise_files = "/home/xjw/data/RIRS_NOISES/noise.scp"


def get_one_noisefile(noise_file=noise_files):
    f = open(noise_file, 'r')
    noise_files = f.readlines()
    num_noises = len(noise_files)
    num = random.randint(0, num_noises-1)
    noise_file = noise_files[num].split()[0]
    return noise_file


np.seterr(divide='ignore', invalid='ignore')

#v = vad.VoiceActivityDetector(speech_energy_threshold = 0.35)
def load_audio(filename, start=0, stop=None, resample=True):#
    sr = SAMPLE_RATE
    y, sr = sf.read(filename, start=start, stop=stop, dtype='float32', always_2d=True)
    y = np.squeeze(y)
    return y, sr

def add_noise(audio_path, noise_path, percent=0.5, sr=16000):
    src, sr = librosa.load(audio_path, sr=sr)
    src_noise, sr = librosa.load(noise_path, sr=sr)
    #print(len(src), len(src_noise))
    if len(src) > len(src_noise):
        n = int(len(src)/len(src_noise))
        src_noise = src_noise.repeat(n+1)
    flag = random.randint(0, len(src_noise) - len(src))
    src_noise = src_noise[flag: flag+len(src)]
    percent = 0.002*random.randint(1,5)
    src = src + percent * src_noise
    S = librosa.core.stft(src, n_fft=N_FFT, hop_length=HOP_LEN, window=hamming)#
    feature, _ = librosa.magphase(S)
    return feature


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None, target_transform=None):
        classes, class_to_idx = self._find_classes(root)
        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif', '.wav','.npy']


def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def npy_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        npy = np.load(f)
        return npy



def accimage_loader(path):
    import accimage
    try:
        return accimage.Image(path)
    except IOError:
        # Potentially a decoding problem, fall back to PIL.Image
        return pil_loader(path)


def default_loader(path):
    from torchvision import get_image_backend
    if get_image_backend() == 'accimage':
        return accimage_loader(path)
    else:
        return pil_loader(path)

def npy_loader(path):
    npys = np.zeros((300, 64))
    temp = np.load(path)
    l = temp.shape[0]
    #print(l)
    if l <= 300:
        npys[:l, :] = temp
        npys[l:, :] = temp[:300-l, :]
    else:
        randint = np.random.randint(l - 300)
        npys = temp[randint: randint+300, :]
    mu = np.average(npys)
    sigma = np.std(npys)
    npys = (npys - mu) / max(sigma, 0.01) 
    #npys = (npys - np.min(npys))/(np.max(npys)-np.min(npys))
    return npys

'''
def wav_loader(path):
    #print(path)
    npys = np.zeros((300, 64))
    Temp = audio_processing.mk_MFB(path)
    fs, data = wavfile.read(path)
    data = v.detect_speech(rate = fs, data = data)
    mfcc_data = np.multiply(Temp[:len(data)],np.expand_dims(data[:,1],1))
    temp = mfcc_data[np.all(mfcc_data!=0,1)]
    
    l = temp.shape[0]
    if l <= 300:
        if l < 150:
            temp = Temp
        L = temp.shape[0]
        if L <= 300:
            npys[:L, :] = temp
            npys[L:, :] = temp[:300-L, :]
        else:
            randint = np.random.randint(L - 300)
            npys = temp[randint: randint+300, :]
    else:
        randint = np.random.randint(l - 300)
        npys = temp[randint: randint+300, :]
    mu = np.average(npys)
    sigma = np.std(npys)
    npys = (npys - mu) / max(sigma, 0.01)
    #npys = (npys - np.min(npys))/(np.max(npys)-np.min(npys))
    return npys
'''


def wav_loader(path):
    #print(path)
    npys = np.zeros((300, 64))
    temp = audio_processing.mk_MFB(path)
    l = temp.shape[0]
    if l <= 300:
        npys[:l, :] = temp
        npys[l:, :] = temp[:300-l, :]
    else:
        randint = np.random.randint(l - 300)
        npys = temp[randint: randint+300, :]
    mu = np.average(npys)
    sigma = np.std(npys)
    npys = (npys - mu) / max(sigma, 0.001) 
    #npys = (npys - np.min(npys))/(np.max(npys)-np.min(npys))
    return npys

def wav_loader_stfft(path):
    #print(path)
    #y, sr = load_audio(path)
    num_frame = 300

    #S = librosa.core.stft(y, n_fft=N_FFT, hop_length=HOP_LEN, window=hamming)#
    #feature, _ = librosa.magphase(S)
    noise_file = get_one_noisefile(noise_files)
    noise_file = noise_dir + noise_file
    feature = add_noise(path, noise_file)


    npy = np.log1p(feature)#
    npy = npy.transpose()
    npy = npy[np.newaxis, :, :]
    npy = np.repeat(npy, 3, axis=0)

    l = npy.shape[1]
    if l <= num_frame:
        new = np.zeros((3, num_frame, 161))
        new[:, :l, :] = npy
        new[:, num_frame-l:, :] = npy[:, :l, :]
        npy = new
    else:
        randint = np.random.randint(l - num_frame)
        npy = npy[:, randint: randint+num_frame, :]
    npy = np.swapaxes(npy,1,2)
    mu = np.average(npy)
    sigma = np.std(npy)
    npy = (npy - mu) / max(sigma, 0.001) 
    return npy


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples


class NpyFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png 

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=npy_loader):
        super(NpyFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

class WavFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png 

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=wav_loader):
        super(WavFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

class WavFolder_stfft(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png 

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=wav_loader_stfft):
        super(WavFolder_stfft, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform)
        self.imgs = self.samples

