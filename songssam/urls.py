from django.urls import include,path
from rest_framework import routers
from .views import inference,extract_mfcc

urlppatterns = [
    path('post/',inference)
    path('get/',extract_mfcc)
]