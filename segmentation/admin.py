from django.contrib import admin

from segmentation.models import Dataset, Image, Prediction

# Register your models here.
admin.site.register(Dataset)
admin.site.register(Image)
admin.site.register(Prediction)
