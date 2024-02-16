import io
import os
import zipfile
import json
from PIL import Image as PILImage
import numpy as np

from django.http import HttpResponse
from django.db.models import Q
from django.core.files import File
from rest_framework import viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema, extend_schema_view, OpenApiParameter, OpenApiTypes

from processing.models import Dataset, Picture, Mask
from processing.serializers import DatasetSerializer, PictureSerializer, MaskSerializer, LabelMeSerializer
from processing.permissions import IsOwnerOrReadOnly
from processing.apps import ProcessingConfig
from segmentation import predict, masks


@extend_schema(tags=['datasets'])
@extend_schema_view(
    list=extend_schema(summary='List all datasets'),
    create=extend_schema(summary='Create a new dataset'),
    retrieve=extend_schema(summary='Retrieve a dataset'),
    partial_update=extend_schema(summary='Update a dataset'),
    destroy=extend_schema(summary='Delete a dataset'),
)
class DatasetViewSet(viewsets.ModelViewSet):
    queryset = Dataset.objects.all()
    serializer_class = DatasetSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    http_method_names = ['get', 'post', 'patch', 'delete']

    def create(self, request):
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(owner=request.user)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def get_queryset(self):
        if self.request.user.is_anonymous:
            return self.queryset.filter(public=True)

        return self.queryset.filter(Q(owner=self.request.user) | Q(public=True))


@extend_schema(tags=['pictures'])
@extend_schema_view(
    list=extend_schema(summary='List all images in a dataset'),
    create=extend_schema(summary='Upload a new image'),
    retrieve=extend_schema(summary='Retrieve an image'),
    destroy=extend_schema(summary='Delete an image'),
    bulk_predict=extend_schema(summary='Predict masks for multiple images'),
)
class PictureViewSet(viewsets.ModelViewSet):
    queryset = Picture.objects.all()
    serializer_class = PictureSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    http_method_names = ['get', 'post', 'delete']

    def create(self, request, dataset_pk=None):
        dataset = Dataset.objects.get(pk=dataset_pk)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(dataset=dataset)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    @extend_schema(summary='Delete multiple images', parameters=[OpenApiParameter(name='ids', type=str, location='query', required=True)])
    def bulk_destroy(self, request, dataset_pk=None, format=None):
        ids = [int(id) for id in request.query_params.get('ids').split(',')]
        images = self.queryset.filter(dataset=dataset_pk, id__in=ids)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        images.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    @extend_schema(tags=['masks'], summary='Predict masks for multiple images', parameters=[OpenApiParameter(name='ids', type=str, location='query', required=True)], responses={200: MaskSerializer(many=True)})
    @action(detail=False, methods=['post'], url_path='predict', serializer_class=MaskSerializer)
    def bulk_predict(self, request, dataset_pk=None):
        image_ids = request.query_params.get('ids')

        if image_ids is None:
            return Response({'detail': 'No image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)

        ids = [int(id) for id in image_ids.split(',')]
        images = self.queryset.filter(dataset=dataset_pk, id__in=ids)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        for image in images:
            if hasattr(image, 'mask') and image.mask is not None:
                return Response({'detail': 'Mask already exists for some images.'}, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = int(request.data.get('threshold', 0))

        masks = []
        for image in images:
            mask = predict(ProcessingConfig.model, image.image, area_threshold)

            mask_byte_arr = io.BytesIO()
            mask.save(mask_byte_arr, format='PNG')

            mask = File(mask_byte_arr, name=f'{image.file_basename}_mask.png')
            masks.append(Mask(picture=image, image=mask, threshold=area_threshold))

        masks = Mask.objects.bulk_create(masks)

        return Response(MaskSerializer(masks, many=True).data, status=status.HTTP_201_CREATED)

    @extend_schema(tags=['masks'], summary='Delete predictions for multiple images', parameters=[OpenApiParameter(name='ids', type=str, location='query', required=True)])
    @bulk_predict.mapping.delete
    def bulk_destroy_predictions(self, request, dataset_pk=None):
        image_ids = request.query_params.get('ids')

        if image_ids is None:
            return Response({'detail': 'No image ids provided.'}, status=status.HTTP_400_BAD_REQUEST)

        ids = [int(id) for id in image_ids.split(',')]
        images = self.queryset.filter(dataset=dataset_pk, id__in=ids, mask__isnull=False)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        predictions = Mask.objects.filter(picture__dataset=dataset_pk, picture__id__in=ids)
        predictions.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    def get_queryset(self):
        return self.queryset.filter(dataset=self.kwargs['dataset_pk'])


@extend_schema(tags=['masks'])
@extend_schema_view(
    list=extend_schema(summary='List all predictions for an image'),
    create=extend_schema(summary='Predict a mask for an image'),
    retrieve=extend_schema(summary='Retrieve a prediction'),
    partial_update=extend_schema(summary='Update a prediction'),
    destroy=extend_schema(summary='Delete a prediction'),
    create_label_me=extend_schema(summary='Create a mask using LabelMe'),
    export_label_me=extend_schema(summary='Export a mask to LabelMe'),
)
class MaskViewSet(viewsets.ModelViewSet):
    serializer_class = MaskSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly, IsOwnerOrReadOnly]
    queryset = Mask.objects.all()
    http_method_names = ['get', 'post', 'delete', 'patch']

    def list(self, request, dataset_pk=None, image_pk=None):
        queryset = self.queryset.filter(picture=image_pk)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data, status=status.HTTP_200_OK)

    def create(self, request, dataset_pk=None, image_pk=None):
        original = Picture.objects.get(pk=image_pk)

        if hasattr(original, 'mask') and original.mask is not None:
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = serializer.validated_data['threshold']

        image = predict(ProcessingConfig.model, original.image, area_threshold)

        mask_byte_arr = io.BytesIO()
        image.save(mask_byte_arr, format='PNG')

        mask = File(mask_byte_arr, name=original.filename)
        serializer.save(picture=original, image=mask)
        return Response(serializer.data, status=status.HTTP_201_CREATED)

    def partial_update(self, request, dataset_pk=None, image_pk=None, pk=None):
        original_mask = Mask.objects.get(pk=pk)

        serializer = self.get_serializer(data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = serializer.validated_data['threshold']

        image = predict(ProcessingConfig.model, original_mask.picture.image, area_threshold)
        mask_byte_arr = io.BytesIO()
        image.save(mask_byte_arr, format='PNG')

        original_mask.image = File(mask_byte_arr)
        original_mask.threshold = area_threshold
        new_mask = original_mask.save()

        serializer = self.get_serializer(new_mask)

        return Response(serializer.data, status=status.HTTP_200_OK)

    @extend_schema(request=LabelMeSerializer, responses={200: MaskSerializer})
    @action(detail=False, methods=['post'], url_path='labelme')
    def create_label_me(self, request, dataset_pk=None, image_pk=None):
        serializer = self.get_serializer(data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        original = Picture.objects.get(pk=image_pk)

        if hasattr(original, 'mask') and original.mask is not None:
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        image = PILImage.open(original.image)
        labelme_data = json.loads(serializer.validated_data['json'].read().decode('utf-8'))

        mask = masks.from_labelme(np.array(image), labelme_data)

        mask_byte_arr = io.BytesIO()
        mask.save(mask_byte_arr, format='PNG')
        mask = File(mask_byte_arr, name=original.filename)

        instance = serializer.save(picture=original, image=mask)
        instance_serializer = MaskSerializer(instance)

        return Response(instance_serializer.data, status=status.HTTP_201_CREATED)

    @extend_schema(responses={(200, 'application/octet-stream'): OpenApiTypes.BINARY})
    @action(detail=True, methods=['get'], url_path='labelme')
    def export_label_me(self, request, dataset_pk=None, image_pk=None, pk=None):
        prediction = Mask.objects.get(pk=pk)

        image = PILImage.open(prediction.picture.image)
        mask = PILImage.open(prediction.image)

        labelme_data = masks.from_labelme(prediction.picture.filename, np.array(mask))

        outfile = io.BytesIO()
        with zipfile.ZipFile(outfile, 'w') as zf:
            with zf.open('labelme.json', 'w') as f:
                f.write(labelme_data.encode('utf-8'))

            with zf.open(os.path.basename(prediction.picture.filename), 'w') as f:
                image.save(f, format='PNG')

        response = HttpResponse(outfile.getvalue(), content_type='application/octet-stream')
        response['Content-Disposition'] = f'attachment; filename={prediction.picture.file_basename}_labelme.zip'

        return response

    def get_queryset(self):
        return self.queryset.filter(picture=self.kwargs['image_pk'])

    def get_serializer_class(self):
        if self.action == 'create_label_me':
            return LabelMeSerializer
        return MaskSerializer
