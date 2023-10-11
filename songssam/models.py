from django.db import models

# Create your models here.
class Song:
    def __init__(self, file ,isUser, uuid):
        self.file = file
        self.isUser = isUser
        self.uuid = uuid
