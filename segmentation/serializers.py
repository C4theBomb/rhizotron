from rest_framework.serializers import Serializer, ModelSerializer, ImageField, FloatField, IntegerField, PrimaryKeyRelatedField
from segmentation.models import Dataset, Image, Prediction

class AnalysisSerializer(Serializer):
    image = ImageField(required=False)
    scaling_factor = FloatField(required=False)

class SegmentationSerializer(Serializer):
    image = ImageField(required=False)
    threshold = IntegerField(required=False, default=25)

class DatasetSerializer(ModelSerializer):
    images = PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Dataset
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'owner']
    
    def create(self, validated_data):
        return Dataset.objects.create(**validated_data)
    
class ImageSerializer(ModelSerializer):
    image = ImageField(required=False)

    class Meta:
        model = Image
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'read_only']
    
class PredictionSerializer(ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'read_only']
