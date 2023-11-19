from rest_framework import serializers
from .models import Song,Inference

class SongSerializer(serializers.Serializer):
    fileKey = serializers.CharField(max_length=100)
    isUser = serializers.CharField(max_length=10)
    uuid = serializers.CharField(max_length=100)

    def create(self, validated_data):
        return Song(**validated_data)

    def update(self, instance, validated_data):
        instance.fileKey = validated_data.get('fileKey', instance.fileKey)
        instance.isUser = validated_data.get('isUser', instance.isUser)
        instance.uuid = validated_data.get('uuid',instance.uuid)
        return instance

class InferSerializer(serializers.Serializer):
    wav_path = serializers.CharField(max_length=100)
    fPtrPath = serializers.CharField(max_length=100)
    uuid = serializers.CharField(max_length=100)

    def create(self, validated_data):
        return Inference(**validated_data)

    def update(self, instance, validated_data):
        instance.wav_path = validated_data.get('wav_path', instance.wav_path)
        instance.fPtrPath = validated_data.get('fPtrPath', instance.fPtrPath)
        instance.uuid = validated_data.get('uuid',instance.uuid)
        return instance