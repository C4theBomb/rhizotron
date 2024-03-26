from .local import *

DATABASES = {
    'default': {
        'ENGINE': 'django_prometheus.db.backends.mysql',
        'NAME': 'dev',
        'USER': 'root',
        'PASSWORD': 'passw0rd',
        'HOST': 'mysqldb',
        'PORT': '3306'
    }
}
