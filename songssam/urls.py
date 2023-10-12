from django.urls import path
from .views import inference,extract_mfcc

urlpatterns = [
    path('post/',inference),
]