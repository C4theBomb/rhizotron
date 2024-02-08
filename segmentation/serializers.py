from rest_framework.serializers import Serializer, ModelSerializer, ImageField, FloatField, IntegerField, PrimaryKeyRelatedField, FileField, CharField
from segmentation.models import Dataset, Picture, Mask


class DatasetSerializer(ModelSerializer):
    images = PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Dataset
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'owner']


class PictureSerializer(ModelSerializer):
    image = ImageField()

    class Meta:
        model = Picture
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'dataset']


class MaskSerializer(ModelSerializer):
    class Meta:
        model = Mask
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'image', 'mask', 'picture']
        write_only_fields = ['threshold']
        extra_kwargs = {
            'threshold': {'required': False, 'default': 0},
        }


class AnalysisSerializer(Serializer):
    image = ImageField(required=False)
    scaling_factor = FloatField(required=False)


class SegmentationSerializer(Serializer):
    image = ImageField(required=False)
    threshold = IntegerField(required=False, default=25)


class LabelMeSerializer(ModelSerializer):
    json = FileField()

    class Meta:
        model = Mask
        fields = ['json']
        read_only_fields = ['created', 'updated', 'image', 'mask', 'picture']
        write_only_fields = ['json']
        extra_kwargs = {
            'json': {'required': True},
        }

    def create(self, validated_data):
        return Mask.objects.create(image=validated_data['image'], picture=validated_data['picture'], threshold=0)
