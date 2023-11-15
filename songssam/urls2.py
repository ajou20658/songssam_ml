from django.urls import path
from .views import opencheck1

urlpatterns = [
    path('/',opencheck1),
]