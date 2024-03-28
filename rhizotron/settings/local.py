from .defaults import *

SECRET_KEY = 'django-insecure-k49qkhm&vo1ds7b3p%+_z6bt)!#aind%3x#-$@e8sq&%2kr8mc'

DEBUG = True

ALLOWED_HOSTS = ['*']

DATABASES = {
    'default': {
        'ENGINE': 'django_prometheus.db.backends.sqlite3',
        'NAME': BASE_DIR / 'db.sqlite3',
    }
}

LOGGING = {
    'version': 1,
    'disable_existing_loggers': False,
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'standard'
        },
        'file': {
            'level': 'INFO',
            'class': 'logging.FileHandler',
            'filename': 'logs/run.log',
            'formatter': 'standard'
        },
    },
    'formatters': {
        'standard': {
            'format': '%(asctime)s %(levelname)s %(module)s: %(message)s'
        }
    },
    'root': {
        'handlers': ['console', 'file'],
        'level': 'INFO'
    },
    'loggers': {
        'main': {
            'handlers': ['file'],
            'level': 'INFO',
        }
    }
}
