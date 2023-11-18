from django.db import models

# Create your models here.
class Song:
    def __init__(self, fileKey ,isUser, uuid):
        self.fileKey = fileKey
        self.isUser = isUser
        self.uuid = uuid

class Inference:
    def __init__(self, wav_path,fPtrPath,uuid):
        self.wav_path=wav_path
        self.fPtrPath=fPtrPath
        self.uuid=uuid