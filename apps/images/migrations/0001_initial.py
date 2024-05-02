
from django.conf import settings
from django.db import migrations, models
import django.db.models.deletion
import images.models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
        migrations.swappable_dependency(settings.AUTH_USER_MODEL),
    ]

    operations = [
        migrations.CreateModel(
            name='ImageSet',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('created', models.DateTimeField(auto_now_add=True, verbose_name='Creation Date and Time')),
                ('modified', models.DateTimeField(auto_now=True, verbose_name='Modification Date and Time')),
                ('name', models.CharField(help_text='eg. Delhi-trip, Tajmahal, flowers', max_length=100)),
                ('description', models.TextField()),
                ('dirpath', models.CharField(blank=True, max_length=150, null=True)),
                ('public', models.BooleanField(default=False)),
                ('user', models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='imagesets', to=settings.AUTH_USER_MODEL)),
            ],
        ),
        migrations.CreateModel(
            name='ImageFile',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('name', models.CharField(max_length=150, null=True, verbose_name='Image Name')),
                ('image', models.ImageField(upload_to=images.models.imageset_upload_images_path)),
                ('is_inferenced', models.BooleanField(default=False)),
                ('image_set', models.ForeignKey(help_text='Image Set of the uploading images', on_delete=django.db.models.deletion.CASCADE, related_name='images', to='images.imageset')),
            ],
        ),
        migrations.AddConstraint(
            model_name='imageset',
            constraint=models.UniqueConstraint(fields=('user', 'name'), name='unique_imageset_by_user'),
        ),
    ]