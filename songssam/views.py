from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import HttpResponse, JsonResponse
from rest_framework.response import Response
from rest_framework.parsers import JSONParser
from django.views.decorators.csrf import csrf_exempt
from io import BytesIO
from tqdm import tqdm
import logging
import easydict
import tempfile
import os

from .models import Song
from .serializers import SongSerializer
from .lib import dataset
from .lib import nets
from .lib import spec_utils
from .lib import utils

import librosa
import numpy as np
import soundfile as sf
import torch
import boto3
import logging

# Create your views here.
logger = logging.getLogger(__name__)
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

@csrf_exempt
@api_view(['POST'])
def inference(request):
    serializer = SongSerializer(data = request.data)
    
    if serializer.is_valid():
        logger.info(serializer.data)
        input_resource = serializer.validated_data['file']
        logger.info(request.POST)
        output_dir = serializer.validated_data['output_dir']
        isUser = serializer.validated_data['isUser']
        songId = serializer.validated_data['songId']
        if(serializer.validated_data['isUser']==True):
            userId = serializer.validated_data['userId']
    else:
        logger.info("serializer 오류 발생")
    args = easydict.EasyDict({
        "pretrained_model" : './models/baseline.pth',
        "sr" : 44100,
        "n_fft" : 2048,
        "hop_length" : 1024,
        "batchsize" : 4,
        "cropsize" : 256,
        "postprocess" : 'store_true'
    })
    # p = argparse.ArgumentParser()
    # # p.add_argument('--gpu', '-g', type=int, default=-1)
    # p.add_argument('--pretrained_model', '-P', type=str, default='models/baseline.pth')
    # # p.add_argument('--input', '-i', required=True)
    # p.add_argument('--sr', '-r', type=int, default=44100)
    # p.add_argument('--n_fft', '-f', type=int, default=2048)
    # p.add_argument('--hop_length', '-H', type=int, default=1024)
    # p.add_argument('--batchsize', '-B', type=int, default=4)
    # p.add_argument('--cropsize', '-c', type=int, default=256)
    # # p.add_argument('--tta', '-t', action='store_true')
    # # p.add_argument('--output_dir', '-o', type=str, default="")
    # args = p.parse_args()
    gpu = -1
    
    s3 = boto3.client('s3',aws_access_key_id='AKIATIVNZLQ23AQR4MPK',aws_secret_access_key='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')
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
    try:
        logger.info('loading wave source...')
        with tempfile.NamedTemporaryFile(suffix=".wav",delete=True,dir = './tmp') as temp_file:
            temp_file.write(input_resource.read())
            temp_file.flush()
            temp_file.seek(0)
            X, sr = librosa.load(
                temp_file.name, sr=args.sr, mono=False, dtype=np.float32, res_type='kaiser_fast')
            if X.ndim == 1:
            # mono to stereo
                X = np.asarray([X, X])
            X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
            logger.info('loading wave done')

            sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)

            y_spec, v_spec = sp.separate_tta(X_spec)

            logger.info('validating output directory...')
            if output_dir != "":  # modifies output_dir if theres an arg specified
                output_dir = output_dir.rstrip('/') + '/'
                os.makedirs(output_dir, exist_ok=True)
            logger.info('done')

            print('inverse stft of instruments...', end=' ')
            
            if(isUser == True):
                logger.info('spectrogram_to_wave')
                wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
                logger.info('spectorgram_to_wave done')
                byte_io = BytesIO()
                sf.write(byte_io,wave.T,sr)
                s3.put_object(Body=byte_io.getvalue(),Bucket = "songssam.site",Key="user/"+userId+"_"+songId)
            else:
                logger.info('spectrogram_to_wave')
                wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
                logger.info('spectorgram_to_wave done')
                byte_io = BytesIO()
                logger.info('write start')
                sf.write(byte_io,wave.T,sr)
                s3.put_object(Body=byte_io.getvalue(),Bucket = "songssam.site",Key="inst/"+songId)
                logger.info('write done')
                logger.info('spectrogram_to_wave')
                wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
                logger.info('spectorgram_to_wave done')
                logger.info('write start')
                sf.write(byte_io, wave.T, sr)
                s3.put_object(Body=byte_io.getvalue(),Bucket = "songssam.site",Key="vocal/"+songId)
                logger.info('write done')
            
            
        return JsonResponse({"message":"Success"},status=200)
    except Exception as e:
        error_message = str(e)
        logger.error(error_message)
        return JsonResponse({"error":"error"},status = 411)
    
def extract_mfcc(filepath):
    # MFCC 계산
    y, sr = librosa.load(filepath, sr=None)  # sr=None으로 설정하여 원본 샘플링 속도로 읽음
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)  # MFCC 계산 (20개의 계수)
    feature = []

    # 각 MFCC 계수의 평균 계산
    mfcc_mean = np.mean(mfcc, axis=1)

    # 음역대 특성 추출
    # 각 음역대에 해당하는 MFCC 계수 범위를 선택하여 평균 계산
    very_low_range_feature = np.mean(mfcc_mean[:4])  # 매우 낮은 주파수 대역 (예: 처음 4개 계수의 평균)
    low_range_feature = np.mean(mfcc_mean[4:8])  # 낮은 주파수 대역 (예: 5~8번째 계수의 평균)
    mid_range_feature = np.mean(mfcc_mean[8:12])  # 중간 주파수 대역 (예: 9~12번째 계수의 평균)
    high_range_feature = np.mean(mfcc_mean[12:16])  # 높은 주파수 대역 (예: 13~16번째 계수의 평균)
    very_high_range_feature = np.mean(mfcc_mean[16:])  # 매우 높은 주파수 대역 (예: 17번째 이후 계수의 평균)
    
    feature.append([very_low_range_feature, low_range_feature, mid_range_feature, high_range_feature, very_high_range_feature])

    # 음역대 특성 출력
    print("매우 낮은 주파수 대역 특성:", very_low_range_feature)
    print("낮은 주파수 대역 특성:", low_range_feature)
    print("중간 주파수 대역 특성:", mid_range_feature)
    print("높은 주파수 대역 특성:", high_range_feature)
    print("매우 높은 주파수 대역 특성:", very_high_range_feature)
    
    
    return JsonResponse({'feature':feature})

def separate_audio(input_file, output_dir, gpu_id=0):
    # Build the command for separating audio
    inference(input_file,output_dir)
