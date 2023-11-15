from django.urls import path
from .views import inference
from .views import opencheck
from .views import o

urlpatterns = [
    path('splitter/',inference),
    path('opencheck/',opencheck)
]