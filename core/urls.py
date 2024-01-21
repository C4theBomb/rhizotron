from django.urls import path
from . import views

urlpatterns = [
    path('', views.starter_kit, name='starter_kit'),
]