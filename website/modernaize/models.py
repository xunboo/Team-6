from django.db import models
from django.conf import settings
from django.contrib.auth.models import User

# Create your models here.
#class UploadedImage(models.Model):
#	image = models.ImageField(upload_to='uploaded_images', height_field='image_height', width_field='image_width')
#	image_width = models.PositiveIntegerField(editable=False)
#	image_height = models.PositiveIntegerField(editable=False)
#	uploaded_at = models.DateTimeField(auto_now_add=True)

class ImageMetadata(models.Model):
	filename = models.CharField(max_length=40, primary_key=True)
	should_colorize = models.BooleanField()
	owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True)
	uploaded_at = models.DateTimeField(auto_now_add=True)

class ShareLink(models.Model):
	name = models.CharField(max_length=40, primary_key=True)
	image = models.ForeignKey('ImageMetadata', on_delete=models.CASCADE, to_field='filename')
	owner = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, null=True)
