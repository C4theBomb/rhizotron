from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    owner = models.ForeignKey(User, related_name='datasets', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)
    
    def __str__(self):
        return self.name

class Image(models.Model):
    dataset = models.ForeignKey(Dataset, related_name='images', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    image = models.ImageField(upload_to='images/', editable=False)

    def delete(self):
        self.image.delete()
        super().delete()

class Prediction(models.Model):
    image = models.OneToOneField(Image, related_name='mask', on_delete=models.CASCADE)
    mask = models.ImageField(upload_to='masks/', editable=False)
    threshold = models.IntegerField(default=0)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def delete(self, dataset_pk=None, image_pk=None):
        self.mask.delete()
        super().delete()
