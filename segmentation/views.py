import io
from PIL import Image

from django.http import HttpResponse
from django.db.models import Q
from rest_framework.response import Response
from rest_framework import generics, viewsets, permissions, status, mixins

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from segmentation.processing import root_analysis, threshold, saving
from segmentation.apps import SegmentationConfig
from segmentation.models import Dataset, Image, Prediction
from segmentation.serializers import SegmentationSerializer, AnalysisSerializer, DatasetSerializer, ImageSerializer, PredictionSerializer 


class SegmentationAPIView(generics.GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = SegmentationSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image = serializer.validated_data['image']
        area_threshold = serializer.validated_data['threshold']

        image = Image.open(io.BytesIO(image.read()))

        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = image.unsqueeze(0)
        image = SegmentationConfig.model(image).detach()
        image = image.squeeze(0, 1)
        image = image.numpy().astype(np.uint8)
        image = threshold.threshold_mask(image, area_threshold)

        image = F.to_pil_image(image)
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')

        return HttpResponse(img_byte_arr.getvalue(), content_type='image/png')


class SegmentationLabelmeAPIView(generics.GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = SegmentationSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image = serializer.validated_data['image']
        area_threshold = serializer.validated_data['threshold']
        filename = image.name

        image = Image.open(io.BytesIO(image.read()))

        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = image.unsqueeze(0)
        image = SegmentationConfig.model(image).detach()
        image = image.squeeze(0, 1)
        image = image.numpy().astype(np.uint8)
        image = threshold.threshold_mask(image, area_threshold)

        labelme_data = saving.labelme(filename, image)

        return HttpResponse(labelme_data, content_type='application/json')


class AnalysisAPIView(generics.GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = AnalysisSerializer

    def post(self, request):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        image = serializer.validated_data['image']
        scaling_factor = serializer.validated_data['scaling_factor']
        image = Image.open(io.BytesIO(image.read()))

        image = np.array(image)

        return Response({
            'root_count': root_analysis.find_root_count(image),
            'root_diameter': root_analysis.find_root_diameter(image, scaling_factor),
            'total_root_area': root_analysis.find_total_root_area(image, scaling_factor),
            'total_root_length': root_analysis.find_total_root_length(image, scaling_factor),
            'total_root_volume': root_analysis.find_total_root_volume(image, scaling_factor),
        }, content_type='application/json')


class DatasetViewSet(viewsets.ModelViewSet):
    serializer_class = DatasetSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = Dataset.objects.all()

    def create(self, request):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(owner=request.user)
        return Response(serializer.data)

    def get_queryset(self):
        return self.queryset.filter(Q(owner=self.request.user) | Q(public=True))

class ImageViewSet(viewsets.ModelViewSet):
    serializer_class = ImageSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = Image.objects.all()

    def get_queryset(self):
        return self.queryset.filter(dataset=self.kwargs['dataset_pk'])

class PredictionViewSet(viewsets.GenericViewSet, mixins.ListModelMixin, mixins.CreateModelMixin, mixins.DestroyModelMixin):
    serializer_class = PredictionSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = Prediction.objects.all()

    def list(self):
        queryset = self.queryset.get(image=self.kwargs['image_pk'])
        serializer = self.get_serializer(queryset)
        return Response(serializer.data)
    
    def create(self, request):
        serializer = self.get_serializer(data=request.data)

        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(owner=request.user)
        return Response(serializer.data)

    def delete(self, request):
        queryset = self.queryset.get(image=self.kwargs['image_pk'], owner=request.user)
        queryset.delete()
        return Response(status=status.HTTP_204_NO_CONTENT)
