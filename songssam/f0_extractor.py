from tqdm import tqdm
import numpy as np

from sklearn.cluster import KMeans
from torch.nn import functional as F
from torchaudio.transforms import Resample

import boto3
import pandas as pd
import logging
import glob
import torchcrepe
import yaml
import torch
import librosa
import os
import matplotlib.pyplot as plt


logger = logging.getLogger(__name__)
s3 = boto3.client('s3',aws_access_key_id='AKIATIVNZLQ23AQR4MPK',aws_secret_access_key='nSCu5JPOudC5xxtNnuCePDo+MRdJeXmnJxWQhd9Q')
bucket = "songssam.site"

CREPE_RESAMPLE_KERNEL={}
class F0_Extractor:
    def __init__(self, f0_extractor = 'crepe', sample_rate = 44100, hop_size = 512, f0_min = 65, f0_max = 800):
        
        self.sample_rate = sample_rate
        self.hop_size = hop_size
        self.f0_min = f0_min
        self.f0_max = f0_max
        
        key_str = str(sample_rate)
        if key_str not in CREPE_RESAMPLE_KERNEL:
            CREPE_RESAMPLE_KERNEL[key_str] = Resample(sample_rate, 16000, lowpass_filter_width = 128)
        self.resample_kernel = CREPE_RESAMPLE_KERNEL[key_str]
        
                
    def extract(self, audio, uv_interp = False, device = None, silence_front = 0): # audio: 1d numpy array

        # extractor start time
        n_frames = int(len(audio) // self.hop_size) + 1
                
        start_frame = int(silence_front * self.sample_rate / self.hop_size)
        real_silence_front = start_frame * self.hop_size / self.sample_rate
        audio = audio[int(np.round(real_silence_front * self.sample_rate)) : ]
       
        # extract f0 using crepe        
        
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        resample_kernel = self.resample_kernel.to(device)
        wav16k_torch = resample_kernel(torch.FloatTensor(audio).unsqueeze(0).to(device))
            
        f0, pd = torchcrepe.predict(wav16k_torch, 16000, 80, self.f0_min, self.f0_max, pad=True, model='full', batch_size=512, device=device, return_periodicity=True)
        pd = MedianPool1d(pd, 4)
        f0 = torchcrepe.threshold.At(0.05)(f0, pd)
        f0 = MaskedAvgPool1d(f0, 4)
            
        f0 = f0.squeeze(0).cpu().numpy()
        f0 = np.array([f0[int(min(int(np.round(n * self.hop_size / self.sample_rate / 0.005)), len(f0) - 1))] for n in range(n_frames - start_frame)])
        f0 = np.pad(f0, (start_frame, 0))
        
         
        f0= f0[f0>0]
        return f0
    
        
def f0_feature(centroids):
    features = ['f0_1', 'f0_2', 'f0_3', 'f0_4', 'f0_5', 'f0_6', 'f0_7', 'f0_8']
    data = pd.DataFrame(sorted(centroids), columns=features)
    return data

def extract_centroid(frequency):
    X = np.array(frequency).reshape(-1, 1)
    
    # K-means 클러스터링 알고리즘 적용
    kmeans = KMeans(n_clusters=8, random_state=0).fit(X)

    # 각 클러스터의 센트로이드 값 확인
    centroids = kmeans.cluster_centers_
    
    return kmeans, centroids.astype(int).T

class DotDict(dict):
    def __getattr__(*args):         
        val = dict.get(*args)         
        return DotDict(val) if type(val) is dict else val   

    __setattr__ = dict.__setitem__    
    __delattr__ = dict.__delitem__

def load_config(path_config):
    with open(path_config, "r") as config:
        args = yaml.safe_load(config)
    args = DotDict(args)
    # print(args)
    return args

def start_F0_Extractor(train_path) : #tmp/uuid/slice/아래의 파일들을 탐색
    sample_rate = 44100
    hop_size = 512
    F0_Extractor2 = F0_Extractor(
                            'crepe', 
                            44100, 
                            512, 
                            65, 
                            800)
    preprocess(train_path,F0_Extractor2,sample_rate,hop_size,device='cuda',extensions=['wav'])

def preprocess(path,f0_extractor,sample_rate,hop_size,device,extensions):
    path_srcdir  = os.path.join(path, 'audio') #tmp/uuid/audio
    path_f0dir  = os.path.join(path, 'f0') #tmp/uuid/f0
    logger.info(path_srcdir)
    logger.info(path_f0dir)
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)
    #tmp/uuid/slice/* 파일 이름들
    logger.info(filelist)
    def process(file):
        binfile = file+'.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        #tmp/uuid/audio/파일이름.wav
        path_f0file = os.path.join(path_f0dir, binfile)
        logger.info("이곳에 저장 : "+path_f0file)
        # tmp/uuid/f0/파일이름.npy
        # load audio
        audio, _ = librosa.load(path_srcfile, sr=sample_rate)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        f0 = f0_extractor.extract(audio, uv_interp = False)
        uv = f0 == 0
        if len(f0[~uv]) > 0:
            # interpolate the unvoiced f0
            f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])

            # save npy
            os.makedirs(os.path.dirname(path_f0file), exist_ok=True)
            np.save(path_f0file, f0)
        else:
            logger.info('\n[Error] F0 extraction failed: ' + path_srcfile)
    print('Preprocess the audio clips in :', path_srcdir)
    
    #
    # ' single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)


def MaskedAvgPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    mask = ~torch.isnan(x)
    masked_x = torch.where(mask, x, torch.zeros_like(x))
    ones_kernel = torch.ones(x.size(1), 1, kernel_size, device=x.device)

    # Perform sum pooling
    sum_pooled = F.conv1d(
        masked_x,
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )

    # Count the non-masked (valid) elements in each pooling window
    valid_count = F.conv1d(
        mask.float(),
        ones_kernel,
        stride=1,
        padding=0,
        groups=x.size(1),
    )
    valid_count = valid_count.clamp(min=1)  # Avoid division by zero

    # Perform masked average pooling
    avg_pooled = sum_pooled / valid_count

    return avg_pooled.squeeze(1)

def MedianPool1d(x, kernel_size):
    x = x.unsqueeze(1)
    x = F.pad(x, ((kernel_size - 1) // 2, kernel_size // 2), mode="reflect")
    x = x.squeeze(1)
    x = x.unfold(1, kernel_size, 1)
    x, _ = torch.sort(x, dim=-1)
    return x[:, :, (kernel_size - 1) // 2]

def traverse_dir(
        root_dir,
        extensions,
        amount=None,
        str_include=None,
        str_exclude=None,
        is_pure=False,
        is_sort=False,
        is_ext=True):

    file_list = []
    cnt = 0
    for root, _, files in os.walk(root_dir):
        for file in files:
            if any([file.endswith(f".{ext}") for ext in extensions]):
                # path
                mix_path = os.path.join(root, file)
                pure_path = mix_path[len(root_dir)+1:] if is_pure else mix_path

                # amount
                if (amount is not None) and (cnt == amount):
                    if is_sort:
                        file_list.sort()
                    return file_list
                
                # check string
                if (str_include is not None) and (str_include not in pure_path):
                    continue
                if (str_exclude is not None) and (str_exclude in pure_path):
                    continue
                
                if not is_ext:
                    ext = pure_path.split('.')[-1]
                    pure_path = pure_path[:-(len(ext)+1)]
                file_list.append(pure_path)
                cnt += 1
    if is_sort:
        file_list.sort()
    return file_list

def concatnator(filepath):
    data_list=[]
    for root, dirs, files in os.walk(filepath+"/f0"):
        for filename in files:
            file_path = os.path.join(root,filename)
            data = np.load(file_path)
            data_list.append(data)
    return np.concatenate(data_list,axis=0)

    
