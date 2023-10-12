from django.db import models

# Create your models here.
class Song:
    def __init__(self, fileKey ,isUser, uuid):
        self.fileKey = fileKey
        self.isUser = isUser
        self.uuid = uuid
