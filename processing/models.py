import os

from django.db import models
from django.contrib.auth.models import User
from django_prometheus.models import ExportModelOperationsMixin


class Dataset(ExportModelOperationsMixin('dataset'), models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    owner = models.ForeignKey(
        'auth.User', related_name='datasets', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)


class Picture(ExportModelOperationsMixin('picture'), models.Model):
    dataset = models.ForeignKey(
        'processing.Dataset', related_name='pictures', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/', editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    @property
    def filename(self) -> str:
        return os.path.basename(self.image.name)

    @property
    def filename_noext(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def owner(self) -> User:
        return self.dataset.owner

    @property
    def public(self) -> bool:
        return self.dataset.public


class Mask(ExportModelOperationsMixin('mask'), models.Model):
    picture = models.OneToOneField(
        'processing.Picture', related_name='mask', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='masks/', editable=False)
    threshold = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    root_count = models.IntegerField(default=0)
    average_root_diameter = models.FloatField(default=0)
    total_root_length = models.FloatField(default=0)
    total_root_area = models.FloatField(default=0)
    total_root_volume = models.FloatField(default=0)

    @property
    def filename(self) -> str:
        return os.path.basename(self.image.name)

    @property
    def filename_noext(self) -> str:
        return os.path.splitext(self.filename)[0]

    @property
    def owner(self) -> User:
        return self.picture.owner

    @property
    def public(self) -> bool:
        return self.picture.public


class Model(ExportModelOperationsMixin('model'), models.Model):
    UNET = 'unet'
    RESNET18 = 'resnet18'
    RESNET34 = 'resnet34'
    RESNET50 = 'resnet50'
    RESNET101 = 'resnet101'
    RESNET152 = 'resnet152'

    choices = {
        (UNET, 'UNet'),
        (RESNET18, 'ResNet18'),
        (RESNET34, 'ResNet34'),
        (RESNET50, 'ResNet50'),
        (RESNET101, 'ResNet101'),
        (RESNET152, 'ResNet152'),
    }

    name = models.CharField(max_length=200)
    model_type = models.CharField(max_length=50, choices=choices, default=UNET)
    model_weights = models.FileField(upload_to='models/', editable=False)
    description = models.TextField(blank=True, null=True)
    owner = models.ForeignKey('auth.User', related_name='models', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)
