# Generated by Django 2.0.1 on 2019-06-12 19:30

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('modernaize', '0003_imagemetadata_uploaded_at'),
    ]

    operations = [
        migrations.AddField(
            model_name='imagemetadata',
            name='should_colorize',
            field=models.BooleanField(default=False),
            preserve_default=False,
        ),
    ]
