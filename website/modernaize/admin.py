from django.contrib import admin

# Register your models here.
#from .models import UploadedImage
from .models import ImageMetadata
from .models import ShareLink

admin.site.register(ImageMetadata)
admin.site.register(ShareLink)
