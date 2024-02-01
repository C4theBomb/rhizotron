import os

from django.db import models
from django.contrib.auth.models import User

from django.db.models.signals import post_delete


class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    owner = models.ForeignKey(User, related_name='datasets', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)


class Image(models.Model):
    dataset = models.ForeignKey(Dataset, related_name='images', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    image = models.ImageField(upload_to='images/', editable=False)

    @property
    def filename(self):
        return os.path.basename(self.image.name)


class Prediction(models.Model):
    image = models.OneToOneField(Image, related_name='mask', on_delete=models.CASCADE)
    mask = models.ImageField(upload_to='masks/', editable=False)
    threshold = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    @property
    def filename(self):
        return os.path.basename(self.mask.name)
