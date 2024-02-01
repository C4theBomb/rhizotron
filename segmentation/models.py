import os

from django.db import models
from django.contrib.auth.models import User

from django.core.files.storage import default_storage
from django.db.models.signals import post_delete
from django.db.models import FileField, ImageField


def file_cleanup(sender, **kwargs):
    for fieldname in sender._meta.get_all_field_names():
        try:
            field = sender._meta.get_field(fieldname)
        except:
            field = None

        if field and (isinstance(field, FileField) or isinstance(field, ImageField)):
            inst = kwargs["instance"]
            f = getattr(inst, fieldname)
            m = inst.__class__._default_manager
            if (
                hasattr(f, "path")
                and os.path.exists(f.path)
                and not m.filter(
                    **{"%s__exact" % fieldname: getattr(inst, fieldname)}
                ).exclude(pk=inst._get_pk_val())
            ):
                try:
                    default_storage.delete(f.path)
                except:
                    pass


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


post_delete.connect(file_cleanup, sender=Image, dispatch_uid='segmentation.image.file_cleanup')
post_delete.connect(file_cleanup, sender=Prediction, dispatch_uid='segmentation.prediction.file_cleanup')
