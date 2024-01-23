from django.urls import path
from . import views


urlpatterns = [
    path('api/analysis/', views.AnalysisView.as_view(), name='analysis'),
    path('api/segmentation/', views.SegmentationView.as_view(), name='segmentation'),
    path('api/segmentation/labelme/', views.SegmentationLabelmeView.as_view(), name='segmentation_labelme'),
]
