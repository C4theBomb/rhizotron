from .defaults import *

SECRET_KEY = 'django-insecure-k49qkhm&vo1ds7b3p%+_z6bt)!#aind%3x#-$@e8sq&%2kr8mc'

DEBUG = True

ALLOWED_HOSTS = [
    'localhost',
    '127.0.0.1',
]

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}


