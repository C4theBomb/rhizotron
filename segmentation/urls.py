from django.urls import path
from django.urls.conf import include
from rest_framework_nested.routers import SimpleRouter, NestedSimpleRouter

from segmentation.routers import ImageBulkRouter
from . import views

router = SimpleRouter()
router.register('datasets', views.DatasetViewSet)

images_router = ImageBulkRouter(router, 'datasets', lookup='dataset')
images_router.register('images', views.ImageViewSet)

predictions_router = NestedSimpleRouter(images_router, 'images', lookup='image')
predictions_router.register('predictions', views.PredictionViewSet)

app_name = 'segmentation'
urlpatterns = [
    path('api/', include(router.urls)),
    path('api/', include(images_router.urls)),
    path('api/', include(predictions_router.urls)),
    path('api/segmentation/', views.SegmentationAPIView.as_view()),
    path('api/segmentation/analysis/', views.AnalysisAPIView.as_view()),
    path('api/segmentation/labelme/', views.SegmentationLabelmeAPIView.as_view()),

    path('segmentation/', views.SegmentationView.as_view(), name='segmentation'),

    path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    path('datasets/create/', views.DatasetCreateView.as_view(), name='dataset_create'),
    path('datasets/<int:pk>/update/', views.DatasetUpdateView.as_view(), name='dataset_update'),
    path('datasets/<int:pk>/delete/', views.DatasetDeleteView.as_view(), name='dataset_delete'),
    path('datasets/<int:pk>/analytics', views.DatasetDetailView.as_view(), name='dataset_detail'),

    path('datasets/<int:pk>/', views.ImagesListView.as_view(), name='image_list'),
    path('datasets/<int:dataset_pk>/images/create/', views.ImageCreateView.as_view(), name='image_create'),
    path('datasets/<int:dataset_pk>/images/<int:pk>/update/', views.ImageUpdateView.as_view(), name='image_update'),
    path('datasets/<int:dataset_pk>/images/<int:pk>/delete/', views.ImageDeleteView.as_view(), name='image_delete'),
    path('datasets/<int:dataset_pk>/images/<int:pk>/', views.ImageDetailView.as_view(), name='image_detail'),
]
