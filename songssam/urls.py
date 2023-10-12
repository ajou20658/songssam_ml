from django.urls import path
from .views import inference

urlpatterns = [
    path('post/',inference),
]