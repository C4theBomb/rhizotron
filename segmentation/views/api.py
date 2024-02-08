import io
import os
import zipfile
import json
from PIL import Image

from django.http import HttpResponse
from django.db.models import Q
from django.core.files import File
from rest_framework import generics, viewsets, permissions, status
from rest_framework.decorators import action
from rest_framework.response import Response
from drf_spectacular.utils import extend_schema

from segmentation.models import Dataset, Picture, Mask
from segmentation.serializers import DatasetSerializer, PictureSerializer, MaskSerializer, LabelMeSerializer
from segmentation.processing import root_analysis, saving, predict
from segmentation.apps import SegmentationConfig


@extend_schema(tags=['datasets'])
class DatasetViewSet(viewsets.ModelViewSet):
    serializer_class = DatasetSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    http_method_names = ['get', 'post', 'patch', 'delete']

    def create(self, request):
        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        serializer.save(owner=request.user)
        return Response(serializer.data)

    def get_queryset(self):
        if self.request.user.is_anonymous:
            return Dataset.objects.filter(public=True)

        return Dataset.objects.filter(Q(owner=self.request.user) | Q(public=True))


@extend_schema(tags=['pictures'])
class PictureViewSet(viewsets.ModelViewSet):
    serializer_class = PictureSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    http_method_names = ['get', 'post', 'delete']

    def bulk_destroy(self, request, dataset_pk=None, format=None):
        ids = [int(id) for id in request.query_params.get('ids').split(',')]
        images = self.queryset.filter(dataset=dataset_pk, id__in=ids)

        if len(ids) != len(images):
            return Response({'detail': 'Some images do not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        images.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    @action(detail=False, methods=['post'], url_path='predict')
    def bulk_predict(self, request, dataset_pk=None):
        ids = [int(id) for id in request.query_params.get('ids').split(',')]

        images = self.queryset.filter(dataset=dataset_pk, id__in=ids)

        for image in images:
            if hasattr(image, 'mask') and image.mask is not None:
                return Response({'detail': 'Mask already exists for some images.'}, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = request

    @action(detail=False, methods=['delete'], url_path='predict')
    def bulk_destroy_predictions(self, request, dataset_pk=None):
        ids = [int(id) for id in request.query_params.get('ids').split(',')]

        images = self.queryset.filter(dataset=dataset_pk, id__in=ids, prediction__isnull=False)

        if len(ids) != len(images):
            return Response({'detail': 'Some images / predictions do not exist.'}, status=status.HTTP_400_BAD_REQUEST)

        predictions = Mask.objects.filter(image__dataset=dataset_pk, image__id__in=ids)
        predictions.delete()

        return Response(status=status.HTTP_204_NO_CONTENT)

    def get_queryset(self):
        return Picture.objects.filter(dataset=self.kwargs['dataset_pk'])


@extend_schema(tags=['masks'])
class MaskViewSet(viewsets.ModelViewSet):
    serializer_class = MaskSerializer
    permission_classes = [permissions.IsAuthenticatedOrReadOnly]
    queryset = Mask.objects.all()
    http_method_names = ['get', 'post', 'delete', 'patch']

    def list(self, request, dataset_pk=None, image_pk=None):
        queryset = self.queryset.filter(image=image_pk)

        serializer = self.get_serializer(queryset, many=True)
        return Response(serializer.data)

    def create(self, request, dataset_pk=None, image_pk=None):
        original = Image.objects.get(pk=image_pk)

        if hasattr(original, 'mask') and original.mask is not None:
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        serializer = self.get_serializer(data=request.data)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = serializer.validated_data['threshold']

        image = predict.predict(SegmentationConfig.model, original.image, area_threshold)

        mask_byte_arr = io.BytesIO()
        image.save(mask_byte_arr, format='PNG')

        mask = File(mask_byte_arr, name=f'{original.image.name.split(".")[0]}_mask.png')
        serializer.save(image=original, mask=mask)
        return Response(serializer.data)

    def partial_update(self, request, dataset_pk=None, image_pk=None, pk=None):
        prediction = Mask.objects.get(pk=pk)
        serializer = self.get_serializer(prediction, data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        area_threshold = serializer.validated_data['threshold']

        image = predict.predict(SegmentationConfig.model, prediction.image.image, area_threshold)

        mask_byte_arr = io.BytesIO()
        image.save(mask_byte_arr, format='PNG')
        mask = File(mask_byte_arr, name=f'{prediction.image.image.name.split(".")[0]}_mask.png')

        serializer.save(image=prediction.image, mask=mask)
        return Response(serializer.data)

    @action(detail=True, methods=['get'], url_path='labelme')
    def export_label_me(self, request, dataset_pk=None, image_pk=None, pk=None):
        prediction = Mask.objects.get(pk=pk)

        image = Image.open(prediction.image.image)
        mask = Image.open(prediction.mask)

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

    @action(detail=False, methods=['post'], url_path='labelme', serializer_class=LabelMeSerializer)
    def create_label_me(self, request, dataset_pk=None, image_pk=None):
        serializer = self.get_serializer(data=request.data, partial=True)
        if not serializer.is_valid():
            return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

        original = Image.objects.get(pk=image_pk)

        if hasattr(original, 'mask') and original.mask is not None:
            return Response({'detail': 'Prediction already exists for this image.'}, status=status.HTTP_400_BAD_REQUEST)

        image = Image.open(original.image)
        labelme_data = json.loads(serializer.validated_data['json'].read().decode('utf-8'))

        mask = saving.save_new_mask(image, labelme_data)

        mask_byte_arr = io.BytesIO()
        mask.save(mask_byte_arr, format='PNG')
        mask = File(mask_byte_arr, name=f'{original.image.name.split(".")[0]}_mask.png')

        serializer.save(image=original, mask=mask)

        return HttpResponse(labelme_data, content_type='application/json')


# class SegmentationAPIView(generics.GenericAPIView):
#     permission_classes = [permissions.AllowAny]
#     serializer_class = serializers.SegmentationSerializer

#     def post(self, request):
#         serializer = self.get_serializer(data=request.data)

#         if not serializer.is_valid():
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         image = serializer.validated_data['image']
#         area_threshold = serializer.validated_data['threshold']

#         image = predict.predict(SegmentationConfig.model, io.BytesIO(image.read()), area_threshold)

#         img_byte_arr = io.BytesIO()
#         image.save(img_byte_arr, format='PNG')

#         return HttpResponse(img_byte_arr.getvalue(), content_type='image/png')


# class SegmentationLabelmeAPIView(generics.GenericAPIView):
#     permission_classes = [permissions.AllowAny]
#     serializer_class = serializers.SegmentationSerializer

#     def post(self, request):
#         serializer = self.get_serializer(data=request.data)

#         if not serializer.is_valid():
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         image = serializer.validated_data['image']
#         area_threshold = serializer.validated_data['threshold']
#         filename = image.name

#         image = Image.open(io.BytesIO(image.read()))

#         image = predict.predict(SegmentationConfig.model, image, area_threshold)

#         labelme_data = saving.labelme(filename, image)

#         return HttpResponse(labelme_data, content_type='application/json')


# class AnalysisAPIView(generics.GenericAPIView):
#     permission_classes = [permissions.AllowAny]
#     serializer_class = serializers.AnalysisSerializer

#     def post(self, request):
#         serializer = self.get_serializer(data=request.data)

#         if not serializer.is_valid():
#             return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)

#         image = serializer.validated_data['image']
#         scaling_factor = serializer.validated_data['scaling_factor']
#         image = Image.open(io.BytesIO(image.read()))

#         return Response({
#             'root_count': root_analysis.find_root_count(image),
#             'root_diameter': root_analysis.find_root_diameter(image, scaling_factor),
#             'total_root_area': root_analysis.find_total_root_area(image, scaling_factor),
#             'total_root_length': root_analysis.find_total_root_length(image, scaling_factor),
#             'total_root_volume': root_analysis.find_total_root_volume(image, scaling_factor),
#         }, content_type='application/json')
