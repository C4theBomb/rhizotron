import os

from .defaults import *

SECRET_KEY = os.environ.get('SECRET_KEY')

DEBUG = True

ALLOWED_HOSTS = [
    '3.15.232.18',
    '127.0.0.1',
    '0.0.0.0'
]

CSRF_TRUSTED_ORIGINS = [
    'http://3.15.232.18',
    '127.0.0.1',
    '0.0.0.0'
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': os.environ.get('DATABASE_NAME'),
        'USER': os.environ.get('DATABASE_USERNAME'),
        'PASSWORD': os.environ.get('DATABASE_PASSWORD'),
        'HOST': os.environ.get('DATABASE_HOST'),
        'PORT': '3306'
    }
}

STATICFILES_STORAGE = "whitenoise.storage.CompressedStaticFilesStorage"
