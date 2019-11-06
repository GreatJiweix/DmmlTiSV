Deep multi-metric learning for text-independent speaker verification
==== 

By Jiwei Xu, Xinggang Wang, Bin Feng, Wenyu Liu.

This code is a implementation of the experiments on Voxceleb 1 corpus and Voxceleb 2 corpus.

Dependencies
-------

Python 3.6

Pytorch 1.2.0

librosa

numpy

soundfile

scipy

Download Dataset
-------

These two data sets can be downloaded directly from the official website.

[Voxceleb 1 and Voxceleb 2](http://www.robots.ox.ac.uk/~vgg/data/voxceleb/interspeech2019.html)  



Preprocess data

We will first convert the .m4a file to .wav format.

sh convert.sh

Then we extract the features of the speech signal, save it in npy format, or directly end-to-end training.

python convert_wav_to_npy.py


