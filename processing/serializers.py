from rest_framework import serializers
from rest_framework.serializers import ImageField, FloatField, IntegerField, PrimaryKeyRelatedField, FileField, CharField
from processing.models import Dataset, Picture, Mask, Model


class DatasetSerializer(serializers.ModelSerializer):
    pictures = PrimaryKeyRelatedField(many=True, read_only=True)

    class Meta:
        model = Dataset
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'owner', 'pictures']


class PictureSerializer(serializers.ModelSerializer):
    image = ImageField()

    class Meta:
        model = Picture
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'dataset']


class MaskSerializer(serializers.ModelSerializer):
    class Meta:
        model = Mask
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'image', 'mask', 'picture', 'root_count',
                            'average_root_diameter', 'total_root_length', 'total_root_area', 'total_root_volume']
        write_only_fields = ['threshold']
        extra_kwargs = {
            'threshold': {'required': False, 'default': 0},
        }


class LabelMeSerializer(serializers.ModelSerializer):
    json = FileField()

    class Meta:
        model = Mask
        fields = ['json']
        read_only_fields = ['created', 'updated', 'image', 'mask', 'picture']
        write_only_fields = ['json']
        extra_kwargs = {
            'json': {'required': True},
        }

    def create(self, validated_data) -> Mask:
        return Mask.objects.create(image=validated_data['image'], picture=validated_data['picture'], threshold=0)


class ModelSerializer(serializers.ModelSerializer):
    model_weights = FileField()

    class Meta:
        model = Model
        fields = '__all__'
        read_only_fields = ['created', 'updated', 'owner']
        write_only_fields = ['model_weights', 'model_type']
        extra_kwargs = {
            'model_weights': {'required': True},
        }


class AnalysisSerializer(serializers.Serializer):
    image = ImageField(required=False)
    scaling_factor = FloatField(required=False)


class SegmentationSerializer(serializers.Serializer):
    image = ImageField(required=False)
    threshold = IntegerField(required=False, default=25)
