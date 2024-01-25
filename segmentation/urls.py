from django.urls import path
from django.urls.conf import include
from rest_framework_nested.routers import SimpleRouter, NestedSimpleRouter

from . import views

router = SimpleRouter()
router.register('datasets', views.DatasetViewSet)

images_router = NestedSimpleRouter(router, 'datasets', lookup='dataset')
images_router.register('images', views.ImageViewSet)

predictions_router = NestedSimpleRouter(images_router, 'images', lookup='image')
predictions_router.register('predictions', views.PredictionViewSet)

urlpatterns = [
    path('api/', include(router.urls)),
    path('api/', include(images_router.urls)),
    path('api/', include(predictions_router.urls)),
    path('api/segmentation/', views.SegmentationAPIView.as_view()),
    path('api/segmentation/analysis/', views.AnalysisAPIView.as_view()),
    path('api/segmentation/labelme/', views.SegmentationLabelmeAPIView.as_view()),
]
