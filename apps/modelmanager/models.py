from django.conf import settings
from django.core.validators import FileExtensionValidator
from django.db import models
from django.utils.translation import gettext_lazy as _

from config.models import CreationModificationDateBase

User = settings.AUTH_USER_MODEL

def model_upload_path(instance, filename):
    return f'{instance.uploader.username}/ml_models/{instance.name}/{filename}'

def model_classfile_upload_path(instance, filename):
    return f'{instance.uploader.username}/mlclassfiles/{instance.name}/{filename}'

class MLModel(CreationModificationDateBase):
    uploader = models.ForeignKey(User,
                                 on_delete=models.CASCADE,
                                 related_name='mlmodels')
    name = models.CharField(_('Name'),
                            max_length=100,
                            help_text='Name for the machine learning model'
                            )
    pth_file = models.FileField(_('UUpload Model Pt/Pth File'),
                                upload_to=model_upload_path,
                                validators=[FileExtensionValidator(
                                    allowed_extensions=['pt', 'pth']
                                )],
                                help_text='Allowed extensions are: .pt, .pth'
                                )
    class_filename = models.CharField(_('Class FileName'),
                                      max_length=100,
                                      null=True,
                                      help_text='Name for the class file'
                                      )
    class_file = models.FileField(_('Ml Model Classes file'),
                                  upload_to=model_classfile_upload_path,
                                  validators=[FileExtensionValidator(
                                      allowed_extensions=[
                                          'txt', 'TXT', 'names', 'names', 'yaml', 'YAML']
                                  )],
                                  help_text='Ml Model classes file. Allowed extensions are: .txt, .names, .yaml'
                                  )
    description = models.TextField(_("Model's description"))
    version = models.CharField(_('Ml Model Version'),
                               max_length=51,
                               null=True,
                               blank=True,
                               )

    public = models.BooleanField(default=False)