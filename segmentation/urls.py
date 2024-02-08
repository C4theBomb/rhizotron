from django.urls import path
from django.urls.conf import include
from rest_framework_nested.routers import SimpleRouter, NestedSimpleRouter

from segmentation.routers import BulkNestedRouter
from . import views

dataset_router = SimpleRouter()
dataset_router.register('datasets', views.DatasetViewSet, basename='datasets')

image_router = BulkNestedRouter(dataset_router, 'datasets', lookup='dataset')
image_router.register('images', views.PictureViewSet, basename='images')

mask_router = NestedSimpleRouter(image_router, 'images', lookup='image')
mask_router.register('masks', views.MaskViewSet, basename='masks')


app_name = 'segmentation'
urlpatterns = [
    path('api/', include(dataset_router.urls)),
    path('api/', include(image_router.urls)),
    # path('api/', include(mask_router.urls)),
    # path('api/segmentation/', views.SegmentationAPIView.as_view()),
    # path('api/segmentation/analysis/', views.AnalysisAPIView.as_view()),
    # path('api/segmentation/labelme/', views.SegmentationLabelmeAPIView.as_view()),

    #     path('segmentation/', views.SegmentationView.as_view(), name='segmentation'),

    #     path('datasets/', views.DatasetListView.as_view(), name='dataset_list'),
    #     path('datasets/create/', views.DatasetCreateView.as_view(), name='dataset_create'),
    #     path('datasets/<int:pk>/update/', views.DatasetUpdateView.as_view(), name='dataset_update'),
    #     path('datasets/<int:pk>/delete/', views.DatasetDeleteView.as_view(), name='dataset_delete'),
    #     path('datasets/<int:pk>/analytics', views.DatasetDetailView.as_view(), name='dataset_detail'),

    #     path('datasets/<int:pk>/', views.ImagesListView.as_view(), name='image_list'),
    #     path('datasets/<int:dataset_pk>/images/create/', views.ImageCreateView.as_view(), name='image_create'),
    #     path('datasets/<int:dataset_pk>/images/<int:pk>/update/', views.ImageUpdateView.as_view(), name='image_update'),
    #     path('datasets/<int:dataset_pk>/images/<int:pk>/delete/', views.ImageDeleteView.as_view(), name='image_delete'),
    #     path('datasets/<int:dataset_pk>/images/<int:pk>/', views.ImageDetailView.as_view(), name='image_detail'),
]
