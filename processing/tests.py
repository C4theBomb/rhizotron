import os
import io
import shutil
import json

from rest_framework.test import APIRequestFactory, force_authenticate, APITestCase
from rest_framework import reverse
from django.contrib.auth.models import User
from django.core.files import File
from django.test import override_settings
from django.utils.http import urlencode

from PIL import Image as PILImage
import tempfile
from urllib.parse import urlparse

from processing.models import Dataset, Picture, Mask, Model
from processing.views import DatasetViewSet, PictureViewSet, MaskViewSet, ModelViewSet

MEDIA_ROOT = tempfile.mkdtemp()


class TestDatasetViewSet(APITestCase):
    def setUp(self):
        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(
            name='test', description='test', owner=self.user)
        self.client = APIRequestFactory()

    def test_list_endpoint(self) -> None:
        request = self.client.get('datasets/')
        force_authenticate(request, user=self.user)

        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_retrieve_endpoint(self) -> None:
        request = self.client.get(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=self.user)

        view = DatasetViewSet.as_view({'get': 'retrieve'})
        response = view(request, pk=self.dataset.id)
        self.assertEqual(response.status_code, 200)

        expected_data = {
            'id': self.dataset.id,
            'name': self.dataset.name,
            'description': self.dataset.description,
            'owner': self.user.id,
            'public': self.dataset.public
        }
        self.assertDictContainsSubset(expected_data, response.data)

    def test_create_endpoint(self) -> None:
        data = {'name': 'test2', 'description': 'test2'}
        request = self.client.post('datasets/', data)
        force_authenticate(request, user=self.user)

        view = DatasetViewSet.as_view({'post': 'create'})
        response = view(request)
        self.assertEqual(response.status_code, 201)

        self.assertEqual(Dataset.objects.filter(
            pk=response.data['id']).count(), 1)

    def test_update_endpoint(self) -> None:
        data = {'name': 'updated', 'description': 'updated'}
        request = self.client.patch(f'datasets/{self.dataset.id}/', data)
        force_authenticate(request, user=self.user)

        view = DatasetViewSet.as_view({'patch': 'update'})
        response = view(request, pk=self.dataset.id)
        self.assertEqual(response.status_code, 200)

        self.dataset.refresh_from_db()

        self.assertEqual(self.dataset.name, 'updated')
        self.assertEqual(self.dataset.description, 'updated')

    def test_delete_endpoint(self) -> None:
        request = self.client.delete(f'datasets/{self.dataset.id}/')
        force_authenticate(request, user=self.user)

        view = DatasetViewSet.as_view({'delete': 'destroy'})
        response = view(request, pk=self.dataset.id)
        self.assertEqual(response.status_code, 204)

        with self.assertRaises(Dataset.DoesNotExist):
            Dataset.objects.get(id=self.dataset.id)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestPictureViewSet(APITestCase):
    def setUp(self) -> None:
        os.makedirs(MEDIA_ROOT, exist_ok=True)

        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(
            name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.picture = Picture.objects.create(
            dataset=self.dataset, image=File(image_bytes, name='test.png'))
        self.client = APIRequestFactory()

    def tearDown(self) -> None:
        shutil.rmtree(MEDIA_ROOT)

    def test_list_endpoint(self):
        request = self.client.get(f'images/')
        force_authenticate(request, user=self.user)

        view = PictureViewSet.as_view({'get': 'list'})
        response = view(request, dataset_pk=self.dataset.id)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_retrieve_endpoint(self) -> None:
        request = self.client.get(f'images/{self.picture.id}/')
        force_authenticate(request, user=self.user)

        view = PictureViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.dataset.id,
                        pk=self.picture.id)
        self.assertEqual(response.status_code, 200)

        expected_data = {
            'id': self.picture.id,
            'dataset': self.dataset.id
        }
        self.assertDictContainsSubset(expected_data, response.data)

        parsed_url = urlparse(response.data['image'])
        self.assertTrue(parsed_url.scheme in ['http', 'https'])

    def test_create_endpoint(self) -> None:
        image = PILImage.new('RGB', (100, 100), color='red')

        tmp_file = tempfile.NamedTemporaryFile(suffix='.png')
        image.save(tmp_file)
        tmp_file.seek(0)

        data = {'image': tmp_file}
        request = self.client.post(f'images/', data)
        force_authenticate(request, user=self.user)

        view = PictureViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id)
        self.assertEqual(response.status_code, 201)

        self.assertEqual(Picture.objects.filter(
            pk=response.data['id']).count(), 1)

    def test_delete_endpoint(self) -> None:
        request = self.client.delete(f'images/{self.picture.id}/')
        force_authenticate(request, user=self.user)

        view = PictureViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id,
                        pk=self.picture.id)
        self.assertEqual(response.status_code, 204)

        with self.assertRaises(Picture.DoesNotExist):
            Picture.objects.get(id=self.picture.id)

    def test_bulk_destroy_endpoint(self) -> None:
        request = self.client.delete(
            f'images/bulk_destroy/', QUERY_STRING=urlencode({'ids': f'{self.picture.id}'}))
        force_authenticate(request, user=self.user)

        view = PictureViewSet.as_view({'delete': 'bulk_destroy'})
        response = view(request, dataset_pk=self.dataset.id)
        self.assertEqual(response.status_code, 204)

        self.assertEqual(Picture.objects.filter(
            dataset=self.dataset).count(), 0)

    def test_bulk_predict_endpoint(self) -> None:
        request = self.client.post(f'images/bulk_predict/',
                                   data={'threshold': 15},
                                   QUERY_STRING=urlencode({'ids': f'{self.picture.id}'}))
        force_authenticate(request, user=self.user)

        view = PictureViewSet.as_view({'post': 'bulk_predict'})
        response = view(request, dataset_pk=self.dataset.id)
        self.assertEqual(response.status_code, 201)
        self.assertEqual(len(response.data), Mask.objects.filter(
            picture__dataset=self.dataset).count())

    def test_bulk_destroy_predictions_endpoint(self) -> None:
        request = self.client.delete(
            'images/bulk_destroy_predictions/', QUERY_STRING=urlencode({'ids': f'{self.picture.id}'}))
        force_authenticate(request, user=self.user)

        image = PILImage.new('RGB', (100, 100))
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        Mask.objects.create(picture=self.picture, image=File(
            image_bytes, name='test_mask.png'))

        view = PictureViewSet.as_view({'delete': 'bulk_destroy_predictions'})
        response = view(request, dataset_pk=self.dataset.id)
        self.assertEqual(response.status_code, 204)

        self.assertEqual(Mask.objects.filter(
            picture__dataset=self.dataset).count(), 0)


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestMaskViewSet(APITestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username='test', password='test')
        self.dataset = Dataset.objects.create(
            name='test', description='test', owner=self.user)

        image = PILImage.new('RGB', (100, 100), color='red')
        image_bytes = io.BytesIO()
        image.save(image_bytes, format='PNG')

        self.picture = Picture.objects.create(
            dataset=self.dataset, image=File(image_bytes, name='test.png'))
        self.mask = Mask.objects.create(
            picture=self.picture, image=File(image_bytes, name='test_mask.png'))

        self.picture_no_mask = Picture.objects.create(
            dataset=self.dataset, image=File(image_bytes, name='test.png'))
        self.client = APIRequestFactory()

    def test_list_endpoint(self) -> None:
        request = self.client.get(f'masks/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'get': 'list'})
        response = view(request, image_pk=self.picture.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 1)

    def test_retrieve_endpoint(self) -> None:
        request = self.client.get(f'masks/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'get': 'retrieve'})
        response = view(request, dataset_pk=self.dataset.id,
                        image_pk=self.picture.id, pk=self.mask.id)

        self.assertEqual(response.status_code, 200)

        expected_data = {
            'id': self.mask.id,
            'picture': self.picture.id,
            'threshold': self.mask.threshold
        }
        self.assertDictContainsSubset(expected_data, response.data)
        self.assertIn('root_count', response.data)
        self.assertIn('average_root_diameter', response.data)
        self.assertIn('total_root_length', response.data)
        self.assertIn('total_root_area', response.data)
        self.assertIn('total_root_volume', response.data)

        parsed_url = urlparse(response.data['image'])
        self.assertTrue(parsed_url.scheme in ['http', 'https'])

    def test_create_endpoint(self) -> None:
        data = {'threshold': 0}
        request = self.client.post(f'masks/', data)
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'post': 'create'})
        response = view(request, dataset_pk=self.dataset.id,
                        image_pk=self.picture_no_mask.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Mask.objects.filter(
            pk=response.data['id']).count(), 1)

    def test_update_endpoint(self) -> None:
        data = {'threshold': 5}
        request = self.client.patch(f'masks/{self.mask.id}/', data)
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'patch': 'partial_update'})
        response = view(request, dataset_pk=self.dataset.id,
                        image_pk=self.picture.id, pk=self.mask.id)
        self.assertEqual(response.status_code, 200)

        self.mask.refresh_from_db()
        self.assertEqual(self.mask.threshold, 5)

    def test_delete_endpoint(self) -> None:
        request = self.client.delete(f'masks/{self.mask.id}/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'delete': 'destroy'})
        response = view(request, dataset_pk=self.dataset.id,
                        image_pk=self.picture.id, pk=self.mask.id)
        self.assertEqual(response.status_code, 204)

        with self.assertRaises(Mask.DoesNotExist):
            Mask.objects.get(id=self.mask.id)

    def test_create_labelme_endpoint(self) -> None:
        json_data = {
            'shapes': []
        }

        tmp_file = tempfile.NamedTemporaryFile(suffix='.json')
        tmp_file.write(json.dumps(json_data).encode('utf-8'))
        tmp_file.seek(0)

        data = {'json': tmp_file}

        request = self.client.post(f'masks/labelme/', data)
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'post': 'create_labelme'})
        response = view(request, dataset_pk=self.dataset.id,
                        image_pk=self.picture_no_mask.id)

        self.assertEqual(response.status_code, 201)
        self.assertEqual(Mask.objects.filter(
            pk=self.picture_no_mask.id).count(), 1)

    def test_export_labelme_endpoint(self) -> None:
        request = self.client.get(f'masks/{self.mask.id}/labelme/')
        force_authenticate(request, user=self.user)

        view = MaskViewSet.as_view({'get': 'export_labelme'})
        response = view(request, dataset_pk=self.dataset.id,
                        image_pk=self.picture.id, pk=self.mask.id)

        self.assertEqual(response.status_code, 200)
        self.assertEqual(response['Content-Type'], 'application/octet-stream')


@override_settings(MEDIA_ROOT=MEDIA_ROOT)
class TestModelViewSet(APITestCase):
    def setUp(self) -> None:
        self.user = User.objects.create_user(username='test', password='test')
        # self.model = Model.objects.create(name="test_model", model_type=Model.UNET, model_weights=)

    def test_list_endpoint(self) -> None:
        request = self.client.get('datasets/')
        force_authenticate(request, user=self.user)

        view = DatasetViewSet.as_view({'get': 'list'})
        response = view(request)
        self.assertEqual(response.status_code, 200)
        self.assertEqual(len(response.data), 0)
