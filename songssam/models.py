from django.db import models

# Create your models here.
class Song:
    def __init__(self, file, output_dir, isUser, songId, userId, uuid):
        self.file = file
        self.isUser = isUser
        self.uuid = uuid
