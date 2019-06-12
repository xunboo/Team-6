import os, sys, uuid
from io import BytesIO
#sys.path.insert(0, os.path.abspath('../upscaling/image-super-resolution/'))
from django.shortcuts import render, redirect, get_object_or_404
from django.contrib.auth import login, authenticate
from django.contrib.auth.decorators import login_required
from django.contrib.auth.forms import UserCreationForm
from azure.storage.blob import BlockBlobService, ContentSettings
from PIL import Image
#import numpy as np
#from ISR.models import RDN

#rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
#rdn.model.load_weights('../upscaling_test/weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
# Not safe to call this function in a multithreaded setting so we need to call it here first
#rdn.model._make_predict_function()
# Create your views here.

from django.http import HttpResponse, HttpResponseRedirect
from django.urls import reverse
from .models import ImageMetadata, ShareLink
from .forms import UploadImageForm

def index(request):
    return render(request, 'modernaize/index.html', {
    	'active_page': 'index'
    	})


def upload(request):
	if request.method == 'POST':
		form = UploadImageForm(request.POST, request.FILES)
		if form.is_valid():
			image_filename = str(uuid.uuid4())
			owner = None
			if request.user.is_authenticated:
				owner = request.user
			# Save image in Azure Blob storage
			img = Image.open(request.FILES['file'])
			#print("Image dimensions are", img.size)
			output_blob = BytesIO()
			img.convert('RGB').save(output_blob, format='JPEG')
			output_blob.seek(0)
			block_blob_service = BlockBlobService(account_name='magnifaistorage', account_key=os.getenv('AZ_STORAGE_KEY'))
			upload_container_name = 'uploads'
			if form.cleaned_data['should_colorize']:
				upload_container_name = 'uploads-colorize'
			block_blob_service.create_blob_from_stream(upload_container_name, image_filename + '.jpg', output_blob,
														content_settings=ContentSettings(content_type='image/jpeg'))
			#print("!!! Uploaded image to blob service.")
			# Create metadata for image in database
			metadata = ImageMetadata(filename=image_filename, should_colorize=form.cleaned_data['should_colorize'], owner=owner)
			#print("!!! Should colorize:", form.cleaned_data['should_colorize'])
			# Create shareable link for image in database
			link = ShareLink(name=str(uuid.uuid4()), image=metadata, owner=owner)
			metadata.save()
			#print("!!! Stored image metadata in database.")
			link.save()
			#print("!!! Stored shareable link data in database.")
			if request.user.is_authenticated:
				return redirect('modernaize:image', image_filename)
			else:
				return redirect('modernaize:share', link.name)
			#form.save()
			#return HttpResponseRedirect(reverse('modernaize:recent'))
			#return HttpResponse("Image uploaded successfully!")
	else:
		form = UploadImageForm()
	return render(request, 'modernaize/upload.html', {
		'form': form,
		'active_page': 'upload'
		})

def recent(request):
	most_recent = None
	if UploadedImage.objects.count() > 0:
		#most_recent = [UploadedImage.objects.latest('uploaded_at')]
		most_recent = UploadedImage.objects.order_by('-uploaded_at')[:3]

	return render(request, 'modernaize/view_recent.html', {
		'recent_images': most_recent,
		'active_page': 'recent'
		})
	#else:
	#	return HttpResponse("No recent images. Perhaps you want to upload one?")

def get_image_info(image_metadata):
	# Ugly hardcoding, but hey, it works :)
	url_root = 'https://magnifaistorage.blob.core.windows.net/'
	state_dict = {}
	block_blob_service = BlockBlobService(account_name='magnifaistorage', account_key=os.getenv('AZ_STORAGE_KEY'))
	orig_container = 'uploads'
	processed_container = 'upscaled'
	if image_metadata.should_colorize:
		orig_container = 'uploads-colorize'
		processed_container = 'upscaled-colorized'
	if block_blob_service.exists(orig_container, image_metadata.filename + '.jpg'):
		state_dict['orig_img_url'] = url_root + orig_container + '/' + image_metadata.filename + '.jpg'
		if image_metadata.owner_id:
			state_dict['image_has_owner'] = True
		if block_blob_service.exists(processed_container, image_metadata.filename + '.jpg'):
			state_dict['upscaled_img_url'] = url_root + processed_container + '/' + image_metadata.filename + '.jpg'
		share_link = ShareLink.objects.filter(image=image_metadata).first()
		state_dict['share_name'] = share_link.name
	return state_dict

@login_required
def image(request, image_filename):
	metadata = get_object_or_404(ImageMetadata, filename=image_filename)
	state_dict = get_image_info(metadata)
	if not 'orig_img_url' in state_dict:
		return redirect('/')
	return render(request, 'modernaize/view_image.html', state_dict)

def share(request, shared_name):
	share_data = get_object_or_404(ShareLink, name=shared_name)
	image_metadata = share_data.image
	state_dict = get_image_info(image_metadata)
	if not 'orig_img_url' in state_dict:
		return redirect('/')
	return render(request, 'modernaize/share_image.html', state_dict)

def register(request):
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            username = form.cleaned_data.get('username')
            raw_password = form.cleaned_data.get('password1')
            user = authenticate(username=username, password=raw_password)
            login(request, user)
            return redirect('modernaize:profile')
    else:
        form = UserCreationForm()
    return render(request, 'registration/register.html', {'form': form})

@login_required
def profile(request):
	cur_user = request.user
	most_recent = ImageMetadata.objects.filter(owner=cur_user).order_by('-uploaded_at')[:3]
	state_dict = {'active_page': 'profile'}
	if most_recent:
		state_dict['recent_images'] = most_recent
	return render(request, 'modernaize/profile.html', state_dict)

def upscale(request, image_id):
	image = get_object_or_404(ImageMetadata, pk=image_id)
	
	#pil_img = Image.open(image.image.name)
	#pil_img = pil_img.convert('RGB')
	#lr_img = np.array(pil_img)
	#rdn = RDN(arch_params={'C':6, 'D':20, 'G':64, 'G0':64, 'x':2})
	#rdn.model.load_weights('../upscaling_test/weights/rdn-C6-D20-G64-G064-x2_ArtefactCancelling_epoch219.hdf5')
	#sr_img = rdn.predict(lr_img)
	#pil_img = Image.fromarray(sr_img)
	#pil_img.save('upscaled_images/result.png')
	
	return render(request, 'modernaize/view_upscale.html', {
		'image': image,
		'active_page': 'upscale'
		})