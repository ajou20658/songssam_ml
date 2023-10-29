from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
from tqdm import tqdm
from pydub import AudioSegment
from pydub.silence import detect_silence, detect_nonsilent
from torch.nn import functional as F
from torchaudio.transforms import Resample

import py7zr
import logging
import easydict
import os
import glob
import torchcrepe
import yaml

from .serializers import SongSerializer
from .lib import dataset
from .lib import nets
from .lib import spec_utils
from .lib import utils

import wave
import magic
import librosa
import numpy as np
import soundfile as sf
import torch
import boto3
import logging
import audioread

# Create your views here.
logger = logging.getLogger(__name__)
s3 = boto3.client('s3',aws_access_key_id='AKIATIVNZLQ23AQR4MPK',aws_secret_access_key='nSCu5JPOudC5xxtNnuCePDo+MRdJeXmnJxWQhd9Q')
bucket = "songssam.site"


class Separator(object):
    def __init__(self, model, device, batchsize, cropsize, postprocess=False):
        self.model = model
        self.offset = model.offset
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess

    def _separate(self, X_mag_pad, roi_size):
        X_dataset = []
        patches = (X_mag_pad.shape[2] - 2 * self.offset) // roi_size
        for i in range(patches):
            start = i * roi_size
            X_mag_crop = X_mag_pad[:, :, start:start + self.cropsize]
            X_dataset.append(X_mag_crop)

        X_dataset = np.asarray(X_dataset)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(X_batch).to(self.device)

                pred = self.model.predict_mask(X_batch)

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)

        return y_spec, v_spec

    def separate(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)
        mask = mask[:, :, :n_frame]

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

    def separate_tta(self, X_spec):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        pad_l, pad_r, roi_size = dataset.make_padding(n_frame, self.cropsize, self.offset)
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask = self._separate(X_mag_pad, roi_size)

        pad_l += roi_size // 2
        pad_r += roi_size // 2
        X_mag_pad = np.pad(X_mag, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        X_mag_pad /= X_mag_pad.max()

        mask_tta = self._separate(X_mag_pad, roi_size)
        mask_tta = mask_tta[:, :, roi_size // 2:]
        mask = (mask[:, :, :n_frame] + mask_tta[:, :, :n_frame]) * 0.5

        y_spec, v_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec

def split_audio_silent(y,sr, output_audio_dir):
    # 오디오 파일 로드
    assert isinstance(y,np.ndarray),"y must be a numpy array"
    print(f"Type of y: {type(y)}, Length of y: {len(y)}, Shape of y: {(y.shape)}")
    # STFT 계산
    D = librosa.stft(y)

    # STFT의 크기(에너지) 계산
    magnitude = np.abs(D)

    # 크기가 작은 스펙트로그램 영역을 식별하여 마스크 생성
    threshold = np.mean(magnitude)*0.5  # 임계값 설정 (평균값 사용)
    mask = magnitude < threshold

    # 마스크를 사용하여 조용한 부분 제거 (소리 있는 부분만 남김)
    D_filtered = D * mask

    # ISTFT 수행하여 분리된 음성 신호 얻기 (조용한 부분)
    y_quiet = librosa.istft(D_filtered)

    # 소리 있는 부분을 얻기 위해 조용한 부분을 뺀다.
    y_noisy = y - y_quiet  # 소음 제거하지 않은 원본 음성에서 조용한 부분을 뺀다.
    n_seconds_threshold = 2  # n초 이상 false인 구간을 찾기 위한 임계값 설정
    mask_false_indices = np.where(y_noisy==False)[0]

    time_intervals = np.diff(mask_false_indices)/sr

    silent_segments = []

    start_idx = mask_false_indices[0]

    for i in range(1,len(mask_false_indices)):
        if time_intervals[i-1] >= n_seconds_threshold:
            end_idx = mask_false_indices[i-1]
            silent_segments.append((start_idx,end_idx))
            start_idx = mask_false_indices[i]

    
    end_idx = mask_false_indices[-1]
    if(len(mask_false_indices)>0) and (time_intervals[-1]>=n_seconds_threshold):
        silent_segments.append((start_idx, end_idx))
    long_silent_segments = []
    for segment in silent_segments:
        start_time, end_time = segment
        duration = end_time - start_time
        if duration >= n_seconds_threshold:
            long_silent_segments.append(segment)
    print(silent_segments)
    start = 0
    end=0
    noisy_segments = []
    for segment in silent_segments:
        if(segment == silent_segments[0]):
            continue
        start=segment[0]
        noisy_segments.append((end,start)) # 이전의 end와 이후의 start == noisy한 구간
        end=segment[1]

    noisy_segments.append((end,-1))
    all_data = silent_segments+noisy_segments
    sorted_all_data = sorted(all_data,key=lambda x:x[0])
    for i in range(len(sorted_all_data)):
        start_time = sorted_all_data[i][0]
        end_time = sorted_all_data[i][1]
        if i%2==0: #silece
            segment_quiet = y[start_time:end_time]
            output_filename_quiet = output_audio_dir+"/"+f"{i}_noisy.wav"
            sf.write(output_filename_quiet,segment_quiet,sr)
        if i%2==1:
            segment_noisy = y[start_time:end_time]
            output_filename_noisy = output_audio_dir+"/"+f"{i}y_quiet.wav"
            sf.write(output_filename_noisy,segment_noisy,sr)
    return len(sorted_all_data)

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def folder_to_7z(folder_path,output_dir): #output_dir/compressed.7z
    with py7zr.SevenZipFile(output_dir+'/compressed.7z','w') as archive:
        for filename in os.listdir(folder_path):
            archive.write(filename)
    logger.info("압축 완료")

def extract_7z(file_path,extract_dir):
    with py7zr.SevenZipFile(file_path+'/compressed.7z','r') as archive:
        archive.extractall(extract_dir)
    logger.info("압축 해제 완료")

def detect_file_type(file_path):
    mime = magic.Magic()
    file_type = mime.from_file(file_path)
    logger.info(file_type)
    if(file_type.__contains__("PCM_16")):
        return "PCM_16"
    elif(file_type.__contains__("PCM_24")):
        return "PCM_24"
    elif(file_type.__contains__("PCM_32")):
        return "PCM_32"
    return "Type Err"

def split_audio_slicing(filenum, input_audio_file,output_audio_dir): #input은 경로
    segment_length_ms = 10000
    audio = AudioSegment.from_wav(input_audio_file)
    
    for start_time in range(0,len(audio),segment_length_ms):
        end_time = start_time + segment_length_ms
        segment = audio[start_time:end_time]
        output_file_path = f"{output_audio_dir}/{filenum}_YES.wav"
        segment.export(output_file_path,format="wav")
        logger.info(output_file_path)
        filenum += 1
    logger.info("split complete")
    return filenum

def load_audio_file(file_path, target_sr=None):
    with audioread.audio_open(file_path) as audio:
        sr = audio.samplerate
        audio_data = []
        for frame in audio:
            audio_data.append(frame)
    return librosa.core.audio.__audioread_load(audio_data, target_sr, mono=False),sr

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
        
         
        # interpolate the unvoiced f0 
        if uv_interp:
            uv = f0 == 0
            if len(f0[~uv]) > 0:
                f0[uv] = np.interp(np.where(uv)[0], np.where(~uv)[0], f0[~uv])
            f0[f0 < self.f0_min] = self.f0_min
        return f0

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
def preprocess(path,f0_extractor,sample_rate,hop_size,device,extensions):
    path_srcdir  = os.path.join(path, 'audio')
    path_f0dir  = os.path.join(path, 'f0')
    
    # list files
    filelist =  traverse_dir(
        path_srcdir,
        extensions=extensions,
        is_pure=True,
        is_sort=True,
        is_ext=True)
    
    def process(file):
        binfile = file+'.npy'
        path_srcfile = os.path.join(path_srcdir, file)
        path_f0file = os.path.join(path_f0dir, binfile)
        
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
            print('\n[Error] F0 extraction failed: ' + path_srcfile)
    print('Preprocess the audio clips in :', path_srcdir)
    
    #
    # ' single process
    for file in tqdm(filelist, total=len(filelist)):
        process(file)
    
def start_F0_Extractor(train_path) :
    _, sample_rate = librosa.load(train_path, sr=None)
    hop_size = 512 * sample_rate / 44100
    
    F0_Extractor = F0_Extractor(
                            'crepe', 
                            44100, 
                            512, 
                            65, 
                            800)
    preprocess(train_path,F0_Extractor,sample_rate,hop_size,device='cuda',extensions=['wav'])

@csrf_exempt
@api_view(['POST'])
def inference(request):
    serializer = SongSerializer(data = request.data)
    root = os.path.abspath('.')
    tmp_path = root+"/songssam/tmp"
    # logger.info(serializer)
    if serializer.is_valid():
        fileKey = serializer.validated_data['fileKey']
        isUser = serializer.validated_data['isUser']
        uuid = serializer.validated_data['uuid']
    else:
        logger.info("serializer 오류 발생")
    
    if not os.path.exists(tmp_path+"/"+str(uuid)):
        os.makedirs(tmp_path+"/"+str(uuid))
    else:
        logger.info("folder already exists")
    tmp_path=tmp_path+"/"+str(uuid)
    filename=tmp_path+"/mp3"
    s3.download_file(bucket,fileKey,filename)
    try:
        
        # input_resource = wave.open(filename,'rb')
        args = easydict.EasyDict({
            "pretrained_model" : root+'/songssam/models/baseline.pth',
            "sr" : 44100,
            "n_fft" : 2048,
            "hop_length" : 1024,
            "batchsize" : 4,
            "cropsize" : 256,
            "postprocess" : 'store_true'
        })
        gpu = 0
        
        print('loading model...', end=' ')
        device = torch.device('cpu')
        model = nets.CascadedNet(args.n_fft, 32, 128)
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))
        if gpu >= 0:
            if torch.cuda.is_available():
                device = torch.device('cuda:{}'.format(gpu))
                model.to(device)
            elif torch.backends.mps.is_available() and torch.backends.mps.is_built():
                device = torch.device('mps')
                model.to(device)
        logger.info('model done')
        
        X, sr = librosa.load(
            filename, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
        
        
        if X.ndim == 1:
        # mono to stereo
            X = np.asarray([X, X])
        logger.info(X.ndim)
        audio_format2 = detect_file_type(filename)
        logger.info(audio_format2)
        # logger.info("file data, sr extract...")
        # if(audio_format2=="Type Err"):

        #     return JsonResponse({"error":"wrong type error"},status = 411)
        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        logger.info(X_spec.dtype)

        sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

        y_spec, v_spec = sp.separate_tta(X_spec)
        logger.info(y_spec.ndim)
        logger.info(y_spec.dtype)
        print('inverse stft of instruments...', end=' ')
        
        if(isUser!="true"):
            logger.info('MR loading...')
            waveT = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
            
            MR_file_path = tmp_path+"/Mr.wav"
            sf.write(MR_file_path,waveT.T,sr,subtype = 'PCM_16',format='WAV')
            
            logger.info(MR_file_path)
            logger.info("위 경로에 MR 저장완료")
            s3_key = "inst/"+str(uuid)
            s3.upload_file(MR_file_path,Bucket = "songssam.site",Key=s3_key)

        ##########################################################
        logger.info('보컬 loading...')
        waveT = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
        output_file_path = tmp_path+"/Vocal.wav"

        
        sf.write(output_file_path,waveT.T,sr,subtype = 'PCM_16',format='WAV')
        logger.info("위 경로에 MR 저장완료")
        y, sr = librosa.load(output_file_path)
        split_path = tmp_path+"/silent_noise" # "/home/ubuntu/git/songssam_ml/songssam/tmp/uuid/silent_noise"
        #######tmp/uuid/silent_noise폴더 생성
        if not os.path.exists(tmp_path+"/"+str(uuid)):
            os.makedirs(tmp_path+"/"+str(uuid))
        else:
            logger.info("folder already exists")
        ####################################
        FileCount = split_audio_silent(y,sr,split_path)#음성 빈곳과 채워진 곳 분리
        
        ##음성 빈 곳은 두고, 채워진 곳은 10초씩 분리하기, 파일이름 어떻게 해야되지
        ##파일 {No}_YES,{No}_No가 반복됨
        logger.info(FileCount)
        ##tmp_path/uuid/silent_noise 폴더 안의 파일을 리스트로 가져옴
        os.remove(tmp_path+"/mp3") #원본 mp3파일 삭제
        file_list = glob.glob(split_path+'/*')
        logger.info(file_list)
        name = file_list[0].split(split_path)
        sname = name[1].split("_")
        logger.info(sname)##첫번째 파일의 이름을 _ 기준으로 분리하였을때 Yes인지 No인지 확인


        if(sname[0]=="/quite"):#quiet
            logger.info("yes")
            filenum=0
            for i in range(FileCount):
                tmp_file = file_list[i]
                filenum = split_audio_slicing(filenum,tmp_file)
        else:
            #noisy
            logger.info("No")

        
        #압축파일 전송
        folder_to_7z(tmp_path+"/slient_noise",tmp_path)
        compressed_file=tmp_path+"/compressed.7z"
        s3_key = "vocal/"+str(uuid)
        s3.upload_file(compressed_file,Bucket = "songssam.site",Key=s3_key)
        
        #silent_noise폴더 비우기
        delete_files_in_folder(tmp_path+"/slient_noise")
        #tmp폴더 비우기
        delete_files_in_folder(tmp_path)
        return JsonResponse({"message":"Success"},status=200)
    except Exception as e:
        error_message = str(e)
        logger.error(error_message)
        return JsonResponse({"error":"error"},status = 411)
    # finally:
    #     input_resource.close()

