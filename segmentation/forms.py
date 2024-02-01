from django.forms import ModelForm

from segmentation.models import Dataset, Image, Prediction


class ImageUploadForm(ModelForm):
    class Meta:
        model = Image
        fields = ['image']
