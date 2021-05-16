from django.urls import path
from .views import TextToEmotion, PicToEmotion

urlpatterns = [
    path('TextToEmo/', TextToEmotion, name="TextToEmotion"),
    path('PicToEmo/', PicToEmotion, name="PicToEmotion")
]
