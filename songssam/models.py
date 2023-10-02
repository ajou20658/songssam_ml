from django.db import models

# Create your models here.
class Song:
    def __init__(self, file, output_dir, isUser, songId, userId):
        self.file = file
        self.output_dir = output_dir
        self.isUser = isUser
        self.songId = songId
        self.userId = userId
