
from django.urls import reverse
from django.http import HttpResponse
from django.db.models import Q
from django.views.generic import ListView, DetailView, CreateView, UpdateView, DeleteView, TemplateView

from segmentation.models import Dataset, Picture, Mask


class DatasetCreateView(CreateView):
    model = Dataset
    fields = ['name', 'description', 'public']

    def form_valid(self, form):
        form.instance.owner = self.request.user
        return super().form_valid(form)

    def get_success_url(self):
        return reverse('segmentation:dataset_detail', kwargs={'pk': self.object.pk})


class DatasetUpdateView(UpdateView):
    model = Dataset
    fields = ['name', 'description', 'public']

    def get_queryset(self):
        return self.model.objects.filter(owner=self.request.user)

    def get_success_url(self):
        return reverse('segmentation:dataset_detail', kwargs={'pk': self.object.pk})


class DatasetDeleteView(DeleteView):
    model = Dataset

    def get_queryset(self):
        return self.model.objects.filter(owner=self.request.user)

    def get_success_url(self):
        return reverse('segmentation:dataset_list')


class DatasetListView(ListView):
    model = Dataset
    ordering = ['-created']
    paginate_by = 5

    def get_queryset(self):
        if self.request.user.is_anonymous:
            return self.model.objects.filter(public=True)

        return self.model.objects.filter(Q(owner=self.request.user) | Q(public=True))


class DatasetDetailView(DetailView):
    model = Dataset


class ImagesListView(ListView):
    model = Picture
    ordering = ['-created']
    paginate_by = 5

    def get_queryset(self):
        dataset = Dataset.objects.get(pk=self.kwargs['pk'])
        return self.model.objects.filter(dataset=dataset)


class ImageDetailView(DetailView):
    model = Picture


class ImageCreateView(CreateView):
    model = Picture
    fields = ['name', 'description', 'image']

    def form_valid(self, form):
        form.instance.dataset = Dataset.objects.get(pk=self.kwargs['dataset_pk'])
        return super().form_valid(form)

    def get_success_url(self):
        return reverse('segmentation:image_list', kwargs={'pk': self.object.dataset.pk})


class ImageUpdateView(UpdateView):
    model = Picture

    def get_queryset(self):
        return self.model.objects.filter(dataset__owner=self.request.user)

    def get_success_url(self):
        return reverse('segmentation:image_detail', kwargs={'dataset_pk': self.object.dataset.pk, 'pk': self.object.pk})


class ImageDeleteView(DeleteView):
    model = Picture

    def get_queryset(self):
        return self.model.objects.filter(dataset__owner=self.request.user)

    def get_success_url(self):
        return reverse('segmentation:image_list', kwargs={'pk': self.object.dataset.pk})


class SegmentationView(TemplateView):
    template_name = 'segmentation/segmentation.html'

    def post(self, request):
        return HttpResponse('Whats up')
