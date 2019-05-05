# Generated by Django 2.2 on 2019-05-05 01:54

from django.db import migrations, models


class Migration(migrations.Migration):

    replaces = [('modernaize', '0001_initial'), ('modernaize', '0002_auto_20190505_0118')]

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='UploadedImage',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('image', models.ImageField(height_field='image_height', upload_to='uploaded_images', width_field='image_width')),
                ('image_width', models.PositiveIntegerField(editable=False)),
                ('image_height', models.PositiveIntegerField(editable=False)),
                ('uploaded_at', models.DateTimeField(auto_now_add=True)),
            ],
        ),
    ]
