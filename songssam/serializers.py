from rest_framework import serializers

class ModelSerializer(serializers.ModelSerializer):
    class Meta:
        file = serializers.FileField()
        output_dir = serializers.CharField(max_length=100)
        isUser = serializers.CharField(max_length=10)
        songId = serializers.IntegerField()
        userId = serializers.IntegerField()