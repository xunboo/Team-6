from django.urls import path
from . import views

app_name = 'modernaize'
urlpatterns = [
    path('', views.index, name='index'),
    path('upload', views.upload, name='upload'),
    path('image/<str:image_filename>/', views.image, name='image'),
    path('share/<str:shared_name>/', views.share, name='share'),
    #path('recent', views.recent, name='recent'),
    #path('upscale/<int:image_id>/', views.upscale, name='upscale'),
    path('accounts/profile/', views.profile, name='profile'),
    path('accounts/register/', views.register, name='register'),
]
