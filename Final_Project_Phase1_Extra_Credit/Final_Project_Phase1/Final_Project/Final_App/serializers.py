from .models import FileModel, AlgorithmModel, AnalyticModel
from rest_framework import serializers

class FileSerializer(serializers.ModelSerializer):
    class Meta:
        model = FileModel
        fields = '__all__'

class AlgorithmSerializer(serializers.ModelSerializer):
    class Meta:
        model = AlgorithmModel
        fields = '__all__'


class AnalyticSerializer(serializers.ModelSerializer):
    class Meta:
        model = AnalyticModel
        fields = '__all__'


