from rest_framework import serializers
from .models import Song

class SongSerializer(serializers.Serializer):
    file = serializers.FileField()
    output_dir = serializers.CharField(max_length=100)
    isUser = serializers.CharField(max_length=10)
    songId = serializers.IntegerField()
    userId = serializers.IntegerField(required=False)

    def create(self, validated_data):
        return Song(**validated_data)

    def update(self, instance, validated_data):
        instance.file = validated_data.get('file', instance.file)
        instance.isUser = validated_data.get('isUser', instance.isUser)
        instance.songId = validated_data.get('songId', instance.songId)
        instance.userId = validated_data.get('userId', instance.userId)
        instance.uuid = validated_data.get('uuid',instance.uuid)
        return instance
