from django.shortcuts import render
# Create your views here.

from django.http import HttpResponse
from .models import UploadedImage
from .forms import UploadImageForm

def index(request):
    return render(request, 'index.html')


def upload(request):
	if request.method == 'POST':
		form = UploadImageForm(request.POST, request.FILES)
		if form.is_valid():
			form.save()
			return HttpResponse("Image uploaded successfully!")
	else:
		form = UploadImageForm()
	return render(request, 'upload.html', {
		'form': form
		})

def recent(request):
	if UploadedImage.objects.count() > 0:
		most_recent = UploadedImage.objects.latest('uploaded_at')
		return render(request, 'view_recent.html', {
			'recent_images': [most_recent]
			})
	else:
		return HttpResponse("No recent images. Perhaps you want to upload one?")
