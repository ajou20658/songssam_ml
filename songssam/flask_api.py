import io
import logging
import torch
import numpy as np
import slicer
import soundfile as sf
import librosa
from flask import Flask, request,Response
from flask_cors import CORS
from pydub import audio_segment
from pydub import AudioSegment

from ddsp.vocoder import load_model, F0_Extractor, Volume_Extractor, Units_Encoder
from ddsp.core import upsample
from enhancer import Enhancer

import oss
import boto3

app = Flask(__name__)

logging.getLogger("numba").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)
s3 = boto3.client('s3',aws_access_key_id='AKIATIVNZLQ23AQR4MPK',aws_secret_access_key='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')
bucket = "songssam.site"

checkpoint_path = "exp/multi_speaker/model_300000.pt"



if __name__ == "__main__":
    # ddsp-svc下只需传入下列参数。
    # 对接的是串串香火锅大佬https://github.com/zhaohui8969/VST_NetProcess-。建议使用最新版本。
    # flask部分来自diffsvc小狼大佬编写的代码。
    # config和模型得同一目录。
    

    # 此处与vst插件对应，端口必须接上。
    app.run(port=6844, host="0.0.0.0", debug=False, threaded=False)
