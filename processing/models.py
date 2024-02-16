import os

from django.db import models


class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField(blank=True, null=True)
    owner = models.ForeignKey('auth.User', related_name='datasets', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)


class Picture(models.Model):
    dataset = models.ForeignKey('processing.Dataset', related_name='pictures', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='images/', editable=False)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    @property
    def filename(self):
        return os.path.basename(self.image.name)

    @property
    def file_basename(self):
        return os.path.splitext(self.filename)[0]

    @property
    def owner(self):
        return self.dataset.owner

    @property
    def public(self):
        return self.dataset.public


class Mask(models.Model):
    picture = models.OneToOneField('processing.Picture', related_name='mask', on_delete=models.CASCADE)
    image = models.ImageField(upload_to='masks/', editable=False)
    threshold = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    @property
    def filename(self):
        return os.path.basename(self.image.name)

    @property
    def file_basename(self):
        return os.path.splitext(self.filename)[0]

    @property
    def owner(self):
        return self.picture.owner

    @property
    def public(self):
        return self.picture.public
