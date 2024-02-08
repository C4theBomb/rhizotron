from django.contrib import admin

from segmentation.models import Dataset, Picture, Mask

# Register your models here.
admin.site.register(Dataset)
admin.site.register(Picture)
admin.site.register(Mask)
