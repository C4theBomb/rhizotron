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
    
class ImageSerializer(ModelSerializer):
    image = ImageField()

    class Meta:
        model = Image
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'dataset']

    
class PredictionSerializer(ModelSerializer):
    class Meta:
        model = Prediction
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'image', 'mask']
        write_only_fields = ['threshold']
        extra_kwargs = {
            'threshold': {'required': False, 'default': 0},
        }
    
    def create(self, validated_data):
        image = validated_data['image']
        threshold = validated_data['threshold']
        mask = validated_data['mask']
        
        return Prediction.objects.create(image=image, threshold=threshold, mask=mask)
