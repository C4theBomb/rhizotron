from django.db import models
from django.contrib.auth.models import User

class Dataset(models.Model):
    name = models.CharField(max_length=200)
    description = models.TextField()
    owner = models.ForeignKey(User, related_name='datasets', on_delete=models.CASCADE)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    public = models.BooleanField(default=False)

    def __unicode__(self):
        return self.name
    
    def __str__(self):
        return self.name

class Image(models.Model):
    dataset = models.ForeignKey(Dataset, related_name='images', on_delete=models.CASCADE)
    name = models.CharField(max_length=200)
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)
    image = models.ImageField(upload_to='images/')

    def __unicode__(self):
        return self.name
    
    def __str__(self):
        return self.name

class Prediction(models.Model):
    image = models.OneToOneField(Image, related_name='mask', on_delete=models.CASCADE)
    mask = models.ImageField(upload_to='masks/')
    created = models.DateTimeField(auto_now_add=True)
    updated = models.DateTimeField(auto_now=True)

    def __unicode__(self):
        return self.name
    
    def __str__(self):
        return self.name
