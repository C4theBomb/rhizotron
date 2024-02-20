from .local import *

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.mysql',
        'NAME': 'dev',
        'USER': 'root',
        'PASSWORD': 'passw0rd',
        'HOST': 'mysqldb',
        'PORT': '3306'
    }
}
