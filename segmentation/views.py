import io
from PIL import Image

from django.http import HttpResponse
from rest_framework.generics import GenericAPIView
from rest_framework.response import Response
from rest_framework import permissions,  status

import numpy as np
import torch
from torchvision.transforms.v2 import functional as F

from segmentation.processing import root_analysis, threshold, saving
from segmentation.apps import SegmentationConfig
from segmentation.serializers import SegmentationSerializer, AnalysisSerializer


class SegmentationView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = SegmentationSerializer
    name = 'segmentation'

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


class SegmentationLabelmeView(GenericAPIView):
    permission_classes = [permissions.AllowAny]
    serializer_class = SegmentationSerializer
    name = 'segmentation'

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


class AnalysisView(GenericAPIView):
    """
    Endpoint for performing analysis on an image.

    This view accepts a POST request with an image and a scaling factor.
    It performs various analysis operations on the image and returns the results.
    """

    permission_classes = [permissions.AllowAny]
    serializer_class = AnalysisSerializer
    name = 'analysis'

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
