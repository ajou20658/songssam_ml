from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
from tqdm import tqdm
from pydub import AudioSegment

import py7zr
import logging
import easydict
import os
import glob

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

tmp_path = "/home/ubuntu/git/songssam_ml/songssam/tmp"


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

def split_audio_silent(input_audio_file, output_audio_dir):
    # 오디오 파일 로드
    audio = AudioSegment.from_file(input_audio_file)

    # 음성이 있는 구간과 없는 구간 감지
    non_silent_ranges = audio.detect_silence(silence_thresh=-40, min_silence_len=5000, seek_step=1000)

    # 음성과 음성 없는 구간 번갈아가면서 저장
    segment_counter = 0
    is_silent_segment = False

    for start, end in non_silent_ranges:
        if is_silent_segment:
            segment = audio[start:end]
            output_file = f"{output_audio_dir}/{segment_counter}_YES.wav"
            segment.export(output_file, format="wav")
        else:
            silent_start = start
            silent_end = end
            segment = audio[silent_start:silent_end]
            output_file = f"{output_audio_dir}/{segment_counter}_NO.wav"
            segment.export(output_file, format="wav")
        is_silent_segment = not is_silent_segment
        segment_counter += 1
    return segment_counter

def delete_files_in_folder(folder_path):
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        try:
            if os.path.isfile(file_path):
                os.remove(file_path)
        except Exception as e:
            print(f"Error deleting {file_path}: {e}")

def folder_to_7z(folder_path,output_dir):
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

@csrf_exempt
@api_view(['POST'])
def inference(request):
    serializer = SongSerializer(data = request.data)
    # logger.info(serializer)
    if serializer.is_valid():
        fileKey = serializer.validated_data['fileKey']
        isUser = serializer.validated_data['isUser']
        uuid = serializer.validated_data['uuid']
    else:
        logger.info("serializer 오류 발생")
    
    # if not os.path.exists(tmp_path+"/"+str(uuid)):
    #     os.makedirs(tmp_path+"/"+str(uuid))
    # else:
    #     logger.info("folder already exists")
    filename=tmp_path+"/"+str(uuid)
    s3.download_file(bucket,fileKey,filename)
    try:
        
        # input_resource = wave.open(filename,'rb')
        args = easydict.EasyDict({
            "pretrained_model" : '/home/ubuntu/git/songssam_ml/songssam/models/baseline.pth',
            "sr" : 44100,
            "n_fft" : 2048,
            "hop_length" : 1024,
            "batchsize" : 4,
            "cropsize" : 256,
            "postprocess" : 'store_true'
        })
        gpu = -1
        
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
            filename, sr=args.sr, mono=False, dtype=np.float64, res_type='kaiser_fast')
        
        
        if X.ndim == 1:
        # mono to stereo
            X = np.asarray([X, X])
        
        audio_format2 = detect_file_type(filename)
        logger.info(audio_format2)
        logger.info("file data, sr extract...")
        # if(audio_format2=="Type Err"):

        #     return JsonResponse({"error":"wrong type error"},status = 411)
        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        logger.info(X_spec.dtype)

        sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

        y_spec, v_spec = sp.separate_tta(X_spec)
        logger.info(y_spec.dtype)
        print('inverse stft of instruments...', end=' ')
        
        if(isUser!="true"):
            logger.info('MR loading...')
            waveT = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
            logger.info('저장중...')
            byte_io = BytesIO()
            sf.write(byte_io,waveT.T,sr,subtype = audio_format2,format='WAV')
            byte_io.seek(0) #포인터 돌려주기

            s3_key = "inst/"+str(uuid)
            s3.put_object(Body=byte_io.getvalue(),Bucket = "songssam.site",Key=s3_key,ContentType = "audio/wav")
            
            byte_io.close()

        ##########################################################
        logger.info('보컬 loading...')
        waveT = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
        logger.info('저장중...')
        output_file_path = str(uuid)+".wav"
        sf.write(output_file_path,waveT.T,sr,subtype = audio_format2,format='WAV')
        split_path = tmp_path+"/silent_noise"
        FileCount = split_audio_silent(output_file_path,split_path)#음성 빈곳과 채워진 곳 분리
        ##음성 빈 곳은 두고, 채워진 곳은 10초씩 분리하기, 파일이름 어떻게 해야되지
        ##파일 {No}_YES,{No}_No가 반복됨
        logger.info(FileCount)
        ##tmppath/uuid/silent_noise 폴더 안의 파일을 리스트로 가져옴
        file_list = glob.glob(split_path+'/*')
        logger.info(file_list)
        
        name = file_list[0].split('_')
        logger.info(name)##첫번째 파일의 이름을 _ 기준으로 분리하였을때 Yes인지 No인지 확인


        if(name[1]=="YES"):
            filenum=0
            for i in range(FileCount):
                tmp_file = file_list[i]
                filenum = split_audio_slicing(filenum,tmp_file)
            logger.info("yes")
        else:
            logger.info("No")

        
        #압축파일 전송
        folder_to_7z(tmp_path+"/slient_noise",tmp_path)
        s3_key = "vocal/"+str(uuid)
        s3.put_object(Body=byte_io.getvalue(),Bucket = "songssam.site",Key=s3_key,ContentType="application/x-7z-compressed")
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
