from django.shortcuts import render
from rest_framework.decorators import api_view
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from tqdm import tqdm
from pydub import AudioSegment
from tempfile import NamedTemporaryFile
import tempfile

from .f0_extractor import start_F0_Extractor, concatnator, f0_feature, extract_centroid

import io
import py7zr
import logging
import easydict
import os

from .ddsp.ddsp.vocoder import load_model,F0_Extractor,Volume_Extractor, Units_Encoder
from .ddsp.ddsp.core import upsample
from .ddsp.enhancer import Enhancer
from django.http import HttpResponse
from .serializers import SongSerializer,InferSerializer
from .lib import dataset
from .lib import nets
from .lib import spec_utils

import magic
import librosa
import numpy as np
import soundfile as sf
import pandas as pd
import torch
import boto3
import logging
import audioread

# Create your views here.
logger = logging.getLogger(__name__)
s3 = boto3.client('s3',aws_access_key_id='AKIATIVNZLQ22ODM76PP',aws_secret_access_key='6qHd4dBq9JF8w1V8UjUzP+xoCjXcB8F8bWrQRntP')
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
    threshold = np.mean(magnitude)*0.1  # 임계값 설정 (평균값 사용)
    mask = magnitude > threshold

    # 마스크를 사용하여 조용한 부분 제거 (소리 있는 부분만 남김)
    D_filtered = D * mask

    # ISTFT 수행하여 분리된 음성 신호 얻기 (조용한 부분)
    y_noisy = librosa.istft(D_filtered)

    sf.write(output_audio_dir+"/Fix_Vocal.wav",y_noisy,sr)
    return threshold

def delete_files_in_folder(folder_path):
    for root, dirs, files in os.walk(folder_path):
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

def folder_to_7z(folder_path,output_dir): #to output_dir
    with py7zr.SevenZipFile(output_dir,'w') as archive:
        for filename in os.listdir(folder_path):
            logger.info("이 파일을 압축"+folder_path+"/"+filename)
            archive.write(folder_path+"/"+filename, filename)
    logger.info("압축 완료 : "+output_dir)

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
        output_file_path = f"{output_audio_dir}/{filenum}.wav"
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

@csrf_exempt
@api_view(['POST'])
def inference(request):
    serializer = SongSerializer(data = request.data)
    if serializer.is_valid():
        fileKey = serializer.validated_data['fileKey']
        isUser = serializer.validated_data['isUser']
        uuid = serializer.validated_data['uuid']
    else:
        logger.info("serializer 오류 발생")
        return JsonResponse({"error":"error"},status = 404)
    root = os.path.abspath('.')
    tmp_path = root+"/songssam/tmp"
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
        
        ####################################
        threshold = split_audio_silent(y,sr,tmp_path)#채워진 곳만 분리
        
        ##tmp_path/uuid/silent_noise 폴더 안의 파일을 리스트로 가져옴
        os.remove(tmp_path+"/mp3") #원본 mp3파일 삭제
        
        filenum=0
        if not os.path.exists(tmp_path+"/raw"):
            os.makedirs(tmp_path+"/raw")
        else:
            logger.info("folder already exists")
        if not os.path.exists(tmp_path+"/f0"):
            os.makedirs(tmp_path+"/f0")
        else:
            logger.info("folder already exists")

        filenum = split_audio_slicing(filenum,tmp_path+"/Fix_Vocal.wav",tmp_path+"/raw")
        logger.info(filenum)
        os.remove(output_file_path)
        os.remove(tmp_path+"/Fix_Vocal.wav")
        if not os.path.exists(tmp_path+"/audio"):
            os.makedirs(tmp_path+"/audio")
        else:
            logger.info("folder already exists")
        filter(tmp_path,threshold)
        
        start_F0_Extractor(tmp_path) #tmp/uuid
        f0 = concatnator(tmp_path)
        _, centroids = extract_centroid(f0)
        data = f0_feature(centroids)
        df_json = data.to_json(orient='records')
        compressed_vocal_file=tmp_path+"/compressed.7z" #/tmp/uuid/compressed.7z
        #압축파일 생성
        folder_to_7z(tmp_path+"/audio",compressed_vocal_file)
            #split_path : tmp/uuid/slice
            #tmp_path : tmp/uuid
        logger.info("압축파일 생성완료")
        # 압축파일 전송
        
        s3_key = "vocal/"+str(uuid)
        s3.upload_file(compressed_vocal_file,Bucket = "songssam.site",Key=s3_key)
        logger.info("vocal압축파일 aws업로드 완료")

        # compressed_f0_file=tmp_path+"/compressed_f0.7z"
        # folder_to_7z(tmp_path+"/f0",compressed_f0_file)
        
        # s3_key = "spect/"+str(uuid)
        # s3.upload_file(compressed_f0_file,Bucket = "songssam.site",Key=s3_key)
        logger.info("f0압축파일 aws업로드 완료")
        #silent_noise폴더 비우기
        # delete_files_in_folder(tmp_path+"/slient_noise")
        logger.info("tmp폴더 비우기")
        delete_files_in_folder(tmp_path)
        logger.info(df_json)
        return JsonResponse({"message":df_json},status=200)
        # send_post_request(data,200,uuid)

    except Exception as e:
        error_message = str(e)
        logger.error(error_message)
        # send_post_request([],848,uuid)
        return JsonResponse({"error":"error"},status = 411)
    
class SvcDDSP:
    def __init__(self, model_path, vocoder_based_enhancer, enhancer_adaptive_key, input_pitch_extractor,
                 f0_min, f0_max, threhold, spk_id, spk_mix_dict, enable_spk_id_cover):
        self.model_path = model_path
        self.vocoder_based_enhancer = vocoder_based_enhancer
        self.enhancer_adaptive_key = enhancer_adaptive_key
        self.input_pitch_extractor = input_pitch_extractor
        self.f0_min = f0_min
        self.f0_max = f0_max
        self.threhold = threhold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.spk_id = spk_id
        self.spk_mix_dict = spk_mix_dict
        self.enable_spk_id_cover = enable_spk_id_cover
        
        # load ddsp model
        self.model, self.args = load_model(self.model_path, device=self.device)
        
        # load units encoder
        if self.args.data.encoder == 'cnhubertsoftfish':
            cnhubertsoft_gate = self.args.data.cnhubertsoft_gate
        else:
            cnhubertsoft_gate = 10
        self.units_encoder = Units_Encoder(
            self.args.data.encoder,
            self.args.data.encoder_ckpt,
            self.args.data.encoder_sample_rate,
            self.args.data.encoder_hop_size,
            cnhubertsoft_gate=cnhubertsoft_gate,
            device=self.device)
        
        # load enhancer
        if self.vocoder_based_enhancer:
            self.enhancer = Enhancer(self.args.enhancer.type, self.args.enhancer.ckpt, device=self.device)

    def infer(self, input_wav, pitch_adjust, speaker_id, safe_prefix_pad_length):
        print("Infer!")
        # load input
        audio, sample_rate = librosa.load(input_wav, sr=None, mono=True)
        if len(audio.shape) > 1:
            audio = librosa.to_mono(audio)
        hop_size = self.args.data.block_size * sample_rate / self.args.data.sampling_rate
        
        # safe front silence
        if safe_prefix_pad_length > 0.03:
            silence_front = safe_prefix_pad_length - 0.03
        else:
            silence_front = 0
            
        # extract f0
        pitch_extractor = F0_Extractor(
            self.input_pitch_extractor,
            sample_rate,
            hop_size,
            float(self.f0_min),
            float(self.f0_max))
        f0 = pitch_extractor.extract(audio, uv_interp=True, device=self.device, silence_front=silence_front)
        f0 = torch.from_numpy(f0).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        f0 = f0 * 2 ** (float(pitch_adjust) / 12)
        
        # extract volume
        volume_extractor = Volume_Extractor(hop_size)
        volume = volume_extractor.extract(audio)
        mask = (volume > 10 ** (float(self.threhold) / 20)).astype('float')
        mask = np.pad(mask, (4, 4), constant_values=(mask[0], mask[-1]))
        mask = np.array([np.max(mask[n : n + 9]) for n in range(len(mask) - 8)])
        mask = torch.from_numpy(mask).float().to(self.device).unsqueeze(-1).unsqueeze(0)
        mask = upsample(mask, self.args.data.block_size).squeeze(-1)
        volume = torch.from_numpy(volume).float().to(self.device).unsqueeze(-1).unsqueeze(0)

        # extract units
        audio_t = torch.from_numpy(audio).float().unsqueeze(0).to(self.device)
        units = self.units_encoder.encode(audio_t, sample_rate, hop_size)
        
        # spk_id or spk_mix_dict
        if self.enable_spk_id_cover:
            spk_id = self.spk_id
        else:
            spk_id = speaker_id
        spk_id = torch.LongTensor(np.array([[spk_id]])).to(self.device)
        
        # forward and return the output
        with torch.no_grad():
            output, _, (s_h, s_n) = self.model(units, f0, volume, spk_id = spk_id, spk_mix_dict = self.spk_mix_dict)
            output *= mask
            if self.vocoder_based_enhancer:
                output, output_sample_rate = self.enhancer.enhance(
                    output, 
                    self.args.data.sampling_rate, 
                    f0, 
                    self.args.data.block_size,
                    adaptive_key = self.enhancer_adaptive_key,
                    silence_front = silence_front)
            else:
                output_sample_rate = self.args.data.sampling_rate

            output = output.squeeze().cpu().numpy()
            return output, output_sample_rate
        
use_vocoder_based_enhancer = True

enhancer_adaptive_key = 0

select_pitch_extractor = 'crepe'

limit_f0_min = 50
limit_f0_max = 1100

threhold = -60

spk_id = 1
enable_spk_id_cover = True

spk_mix_dict = None


@csrf_exempt
@api_view(['GET'])
def voice_change_model(request):
    # 요청에서 쿼리 파라미터를 추출하여 InferSerializer로 변환
    serializer = InferSerializer(data=request.GET)

    # serializer가 유효한지 확인
    if serializer.is_valid():
        f_wave_path = serializer.validated_data["wav_path"]
        f_ptr_path = serializer.validated_data["fPtrPath"]
        uuid = serializer.validated_data["uuid"]
    else:
        logger.info("serializer 오류")
        return HttpResponse({"error":"error"},status=404)
    logger.info(f_wave_path)
    logger.info(f_ptr_path)
    # 결과를 저장할 경로 생성
    if not os.path.exists("exp/"+str(uuid)):
        os.makedirs("exp/"+str(uuid))
    else:
        logger.info("folder already exists")

    

    root = os.path.abspath('.')
    tmp_path = root+"/songssam/tmp2"
    mp3_filename = tmp_path+"/Origin.mp3"

    # S3에서 파일 다운로드
    s3.download_file(bucket,f_wave_path,mp3_filename)
    # 별도의 변수를 io.BytesIO 객체로 초기화(이후에 wav 데이터로 사용됨)
    wav_data = tmp_path+"/Wav.wav"
    
        # 음성 변환에 필요한 매개변수 설정
    args = easydict.EasyDict({
        "pretrained_model" : root+'/songssam/models/baseline.pth',
        "sr" : 44100,
        "n_fft" : 2048,
        "hop_length" : 1024,
        "batchsize" : 4,
        "cropsize" : 256,
        "postprocess" : 'store_true'
    })
    
    # 모델 로드
    print('loading model...', end=' ')
    model = nets.CascadedNet(args.n_fft, 32, 128)
    model.load_state_dict(torch.load(args.pretrained_model))

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

    # mp3 파일 로드
    X, sr = librosa.load(
        mp3_filename, sr=args.sr, mono=True, dtype=np.float32, res_type='kaiser_fast')
    
    # 모노 오디오를 스테레오로 변환
    if X.ndim == 1:
        X = np.asarray([X, X])
    logger.info(X.ndim)

    # 파일 형식 검출
    audio_format2 = detect_file_type(mp3_filename)
    logger.info(audio_format2)

    # 음성 데이터를 스펙트럼으로 변환
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    logger.info(X_spec.dtype)

    # 분리기를 이용하여 스펙트럼 분리
    sp = Separator(model, device, args.batchsize, args.cropsize, args.postprocess)
    y_spec, v_spec = sp.separate_tta(X_spec)


    logger.info(y_spec.ndim)
    logger.info(y_spec.dtype)
    print('inverse stft of instruments...', end=' ')
    

    logger.info('MR loading...')
    waveT = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    
    # MR 파일 저장
    MR_file_path = tmp_path+"/Mr.wav"
    sf.write(MR_file_path,waveT.T,sr,subtype = 'PCM_16',format='WAV')

    ##########################################################
    logger.info('보컬 loading...')
    waveT = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)

    sf.write(wav_data,waveT.T,sr,subtype = 'PCM_16',format='WAV')
   
    
    pt_filename = tmp_path+"/Voice.pt"
    # S3에서 .pt 파일 다운로드
    s3.download_file(bucket,f_ptr_path,pt_filename)

    # 변조 정보 설정
    f_safe_prefix_pad_length = float(0)
    f_pitch_change = float(0) #키값 변경
    int_speak_id = int(0)
    daw_sample = float(44100)

    if enable_spk_id_cover:
        int_speak_id = spk_id

    wav_read = AudioSegment.from_file(wav_data,format="wav")
    input_wav_read=io.BytesIO(wav_read.read())
    svc_model = SvcDDSP(pt_filename, use_vocoder_based_enhancer, enhancer_adaptive_key, select_pitch_extractor,
                        limit_f0_min, limit_f0_max, threhold, spk_id, spk_mix_dict, enable_spk_id_cover)
    
    
    # 모델 추론
    _audio, _model_sr = svc_model.infer(input_wav_read, f_pitch_change, int_speak_id, f_safe_prefix_pad_length)
    
    # 오디오 재샘플링
    tar_audio = librosa.resample(_audio, _model_sr, daw_sample)
    
    # 반환할 오디오 파일 작성
    # out_wav_path = tmp_path+"/generated.wav"
    # sf.write(out_wav_path, tar_audio, daw_sample, format="wav")
    # mp3 = AudioSegment.from_file(out_wav_path,format="wav")

    mr_audio = AudioSegment.from_file(MR_file_path, format="wav")
    tar_audio = AudioSegment.from_mono_audiosegmentstar(tar_audio)

    # 오디오 데이터 합치기
    combined_audio = mr_audio.overlay(tar_audio, positon=0)

    # MP3로 오디오를 내보내고, 바이트로 읽어옵니다
    audio_bytes = combined_audio.export(format='mp3').read()

    # HTTP 응답 객체를 생성하고, 오디오 바이트를 content로 설정합니다
    # response = HttpResponse(content=audio_bytes, content_type='audio/mpeg')
    response = HttpResponse(content=audio_bytes, content_type='audio/mpeg')

    return response
 

def filter(filepath,threshold):
    for root, dirs, files in os.walk(filepath+"/raw"):
        filenum=0
        for filename in files:
            file_path = os.path.join(root, filename)
            try:
                y, sr = librosa.load(file_path)
                D = librosa.stft(y)
                
                # STFT의 크기(에너지) 계산
                magnitude = np.abs(D)

                # 크기가 작은 스펙트로그램 영역을 식별하여 마스크 생성
                np.mean(magnitude)  # 임계값 설정 (평균값 사용)
                if np.mean(magnitude) < threshold :
                    os.remove(file_path)
                    print(f"Deleted: {file_path}")
                else:
                    filenum=filenum+1
                    os.rename(file_path,filepath+f"/audio/{filenum}.wav")
            except Exception as e:
                print(f"Error deleting {file_path}: {e}")

@csrf_exempt
@api_view(['GET'])
def opencheck(request):
    return JsonResponse({"message":"Open"},status=200)