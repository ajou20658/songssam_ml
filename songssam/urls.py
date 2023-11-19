from django.urls import path
from .views import inference
from .views import opencheck
from .views import voice_change_model

urlpatterns = [
    path('splitter/',inference),
    path('opencheck/',opencheck),
    path('voiceChangeModel/',voice_change_model)
]