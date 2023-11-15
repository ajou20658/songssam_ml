from django.urls import path
from .views import opencheck2

urlpatterns = [
    path('/',opencheck2),
]