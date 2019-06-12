from .settings import *

DEBUG = False
STATICFILES_STORAGE = 'storages.backends.azure_storage.AzureStorage'
AZURE_ACCOUNT_NAME = 'magnifaistorage'
AZURE_CONTAINER = 'djangostatic'
if not os.getenv('AZ_STORAGE_KEY').endswith('=='):
    os.environ['AZ_STORAGE_KEY'] += '==' # horrible hack because environment variables are hard
AZURE_ACCOUNT_KEY = os.getenv('AZ_STORAGE_KEY')

DATABASES = {
    'default': {
        'ENGINE': 'django.db.backends.postgresql',
        'NAME': 'postgres',
        'USER': 'psqladmin@magnifai-psql-db',
        'PASSWORD': os.getenv('POSTGRES_ADMIN_PASSWORD'),
        'HOST': 'magnifai-psql-db.postgres.database.azure.com',
        'PORT': '5432',
        'OPTIONS': {'sslmode': 'require'},
    }
}