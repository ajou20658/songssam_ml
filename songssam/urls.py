from django.urls import path
from .views import inference
from .views import opencheck

urlpatterns = [
    path('splitter/',inference),
    path('opencheck/',opencheck)
]