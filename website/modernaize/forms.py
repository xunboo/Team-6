from django import forms
from .models import ImageMetadata

class UploadImageForm(forms.Form):
	#class Meta:
	#	model = UploadedImage
	#	fields = ('image',)
	file = forms.ImageField(label='Your image')
	should_colorize = forms.BooleanField(label='Check this if you also want the image colorized:', required=False)
