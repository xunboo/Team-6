from django.db import models

# Create your models here.
class UploadedImage(models.Model):
	image = models.ImageField(upload_to='uploaded_images', height_field='image_height', width_field='image_width')
	image_width = models.PositiveIntegerField(editable=False)
	image_height = models.PositiveIntegerField(editable=False)
	uploaded_at = models.DateTimeField(auto_now_add=True)
