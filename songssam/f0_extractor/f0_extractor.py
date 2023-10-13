from .ddsp.vocoder import f0_extractor
import librosa
import os
import numpy as np

def process(path,file,sample_rate):
    path_srcdir  = os.path.join(path, 'audio')
    path_f0dir  = os.path.join(path, 'f0')


    path_srcfile = os.path.join(path_srcdir, file)
    binfile = file+'.npy'
    path_f0file = os.path.join(path_f0dir, binfile)
    audio, _ = librosa.load(path_srcfile, sr=sample_rate)
    if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)

    f0 = f0_extractor.extract(audio,uv_interp = False)
    np.save(path_f0file, f0)
