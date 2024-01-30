import io
import os
import zipfile
import json
from PIL import Image

from django.http import HttpResponse
from django.db.models import Q
from django.core.files import File
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView
from rest_framework import generics, viewsets, permissions, status, mixins
from rest_framework.decorators import action
from rest_framework.response import Response

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from segmentation import models, serializers
from segmentation.processing import root_analysis, threshold, saving
from segmentation.apps import SegmentationConfig


class SegmentationAPIView(generics.GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = serializers.SegmentationSerializer

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
    serializer_class = serializers.SegmentationSerializer

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
    serializer_class = serializers.AnalysisSerializer

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
    serializer_class = serializers.DatasetSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = models.Dataset.objects.all()
    http_method_names = ['get', 'post', 'delete']

    def create(self, request):
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(owner=request.user)
        return Response(serializer.data)

    def get_queryset(self):
        return self.queryset.filter(Q(owner=self.request.user) | Q(public=True))


class ImageViewSet(viewsets.ModelViewSet):
    serializer_class = serializers.ImageSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = models.Image.objects.all()
    http_method_names = ['get', 'post', 'delete']

    def create(self, request, dataset_pk=None):
        dataset = models.Dataset.objects.get(pk=dataset_pk)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(dataset=dataset)
        return Response(serializer.data)

    def get_queryset(self):
        return self.queryset.filter(dataset=self.kwargs['dataset_pk'])


class PredictionViewSet(viewsets.ModelViewSet):
    serializer_class = serializers.PredictionSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = models.Prediction.objects.all()
    http_method_names = ['get', 'post', 'delete', 'patch']

    def list(self, request, dataset_pk=None, image_pk=None):
        queryset = self.queryset.filter(image=image_pk)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request, dataset_pk=None, image_pk=None):
        original = models.Image.objects.get(pk=image_pk)

        if hasattr(original, 'mask') and original.mask is not None:
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = serializer.validated_data['threshold']

        image = Image.open(original.image)
        image = np.array(image)
        image = image[:, :, :3]
        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = image.unsqueeze(0)
        image = SegmentationConfig.model(image).detach()
        image = image.squeeze(0, 1)
        image = image.numpy().astype(np.uint8)
        image = threshold.threshold_mask(image, area_threshold)
        image = F.to_pil_image(image)

        mask_byte_arr = io.BytesIO()
        image.save(mask_byte_arr, format='PNG')
        mask = File(mask_byte_arr, name=f'{original.image.name.split(".")[0]}_mask.png')

        serializer.save(image=original, mask=mask)
        return Response(serializer.data)

    def partial_update(self, request, dataset_pk=None, image_pk=None, pk=None):
        prediction = models.Prediction.objects.get(pk=pk)
        serializer = self.get_serializer(prediction, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = serializer.validated_data['threshold']

        image = Image.open(prediction.image.image)
        image = np.array(image)
        image = image[:, :, :3]
        image = F.to_image(image)
        image = F.to_dtype(image, torch.float32, scale=True)

        image = image.unsqueeze(0)
        image = SegmentationConfig.model(image).detach()
        image = image.squeeze(0, 1)
        image = image.numpy().astype(np.uint8)
        image = threshold.threshold_mask(image, area_threshold)
        image = F.to_pil_image(image)

        mask_byte_arr = io.BytesIO()
        image.save(mask_byte_arr, format='PNG')
        mask = File(mask_byte_arr, name=f'{prediction.image.image.name.split(".")[0]}_mask.png')

        serializer.save(image=prediction.image, mask=mask)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='labelme')
    def export_label_me(self, request, dataset_pk=None, image_pk=None, pk=None):
        prediction = models.Prediction.objects.get(pk=pk)

        image = Image.open(prediction.image.image)
        mask = Image.open(prediction.mask)
        mask = np.array(mask)

        labelme_data = saving.labelme(prediction.mask.name, mask)

        outfile = io.BytesIO()
        with zipfile.ZipFile(outfile, 'w') as zf:
            with zf.open('labelme.json', 'w') as f:
                f.write(labelme_data.encode('utf-8'))

            with zf.open(os.path.basename(prediction.image.image.name), 'w') as f:
                image.save(f, format='PNG')

        response = HttpResponse(outfile.getvalue(), content_type='application/octet=stream')
        response['Content-Disposition'] = f'attachment; filename={prediction.mask.name.split(".")[0]}_labelme.zip'

        return response

    @action(detail=False, methods=['post'], url_path='labelme')
    def create_label_me(self, request, dataset_pk=None, image_pk=None):
        serializer = self.get_serializer(data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        if hasattr(models.Image.objects.get(pk=image_pk), 'mask'):
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        original = models.Image.objects.get(pk=image_pk)
        image = Image.open(original.image)
        image = np.array(image)
        labelme_data = json.loads(serializer.validated_data['json'].read().decode('utf-8'))

        mask = saving.save_new_mask(image, labelme_data)
        mask = Image.fromarray(mask)

        mask_byte_arr = io.BytesIO()
        mask.save(mask_byte_arr, format='PNG')
        mask = File(mask_byte_arr, name=f'{original.image.name.split(".")[0]}_mask.png')

        serializer.save(image=original, mask=mask)

        return HttpResponse(labelme_data, content_type='application/json')

    def get_serializer_class(self):
        if (self.action == 'create_label_me' and self.request.method == 'POST'):
            return serializers.LabelMeSerializer
        else:
            return self.serializer_class


class DatasetView(ListView):
    model = models.Dataset
    template_name = 'dataset.html'

    def get_queryset(self):
        return self.model.objects.filter(Q(owner=self.request.user) | Q(public=True))
