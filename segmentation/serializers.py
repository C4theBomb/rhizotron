from rest_framework import serializers

class AnalysisSerializer(serializers.Serializer):
    image = serializers.ImageField(required=False)
    scaling_factor = serializers.FloatField(required=False)

class SegmentationSerializer(serializers.Serializer):
    image = serializers.ImageField(required=False)
    threshold = serializers.IntegerField(required=False, default=25)