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
    name = 'segmentation_labelme'

    def post(self, request):
        """
        Process a POST request to perform image segmentation.

        This method takes in an image and a threshold value from the request data.
        It performs image segmentation using the provided threshold and returns the segmented image data in JSON format.

        Parameters:
        - request: The HTTP request object containing the image and threshold data.

        Returns:
        - HttpResponse: The segmented image data in JSON format.

        Raises:
        - status.HTTP_400_BAD_REQUEST: If the request data is invalid.

        Example Usage:
        ```
        POST /segmentation
        {
            "image": <image_data>,
            "threshold": 0.5
        }
        ```
        """
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
    permission_classes = [permissions.AllowAny]
    serializer_class = AnalysisSerializer
    name = 'analysis'

    """
    API endpoint for performing analysis on an image.

    This view accepts a POST request with an image and a scaling factor.
    It performs various root analysis calculations on the image and returns the results.

    Parameters:
    - image: The image file to be analyzed.
    - scaling_factor: The scaling factor to convert pixel measurements to real-world measurements.

    Returns:
    A JSON response containing the following analysis results:
    - root_count: The number of roots detected in the image.
    - root_diameter: The average diameter of the roots in the image.
    - total_root_area: The total area occupied by the roots in the image.
    - total_root_length: The total length of the roots in the image.
    - total_root_volume: The total volume of the roots in the image.

    Example usage:
    ```
    POST /analysis/
    {
        "image": <image_file>,
        "scaling_factor": 0.1
    }
    ```

    Example response:
    ```
    {
        "root_count": 10,
        "root_diameter": 0.5,
        "total_root_area": 100,
        "total_root_length": 50,
        "total_root_volume": 25
    }
    ```
    """

    def post(self, request):
        """
        API endpoint for processing a POST request.

        This endpoint receives an image and a scaling factor, performs root analysis on the image,
        and returns the results in JSON format.

        Parameters:
        - request: The HTTP request object containing the image and scaling factor.

        Returns:
        - A JSON response containing the following root analysis results:
            - root_count: The number of roots detected in the image.
            - root_diameter: The average diameter of the roots in the image.
            - total_root_area: The total area covered by the roots in the image.
            - total_root_length: The total length of the roots in the image.
            - total_root_volume: The total volume of the roots in the image.
        """
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
