from config.models import CreationModificationDateBase

from django.db import models
from django.utils.translation import gettext_lazy as _

# Create your models here.
class InferencedImage(CreationModificationDateBase):
    orig_image = models.ForeignKey(
        "images.ImageFile",
        on_delete=models.CASCADE,
        related_name="detectedimages",
        help_text="Main Image",
        null=True,
        blank=True
    )

    inf_image_path = models.CharField(max_length=250,
                                      null=True,
                                      blank=True
                                      )

    custom_model = models.ForeignKey("modelmanager.MLModel",
                                     verbose_name="Custom ML Models",
                                     on_delete=models.DO_NOTHING,
                                     null=True,
                                     blank=True,
                                     related_name="detectedimages",
                                     help_text="Machine Learning model for detection",
                                     )
    detection_info = models.JSONField(null=True, blank=True)
    
    AIMODEL_CHOICES = [
        ('vggnet.pt','vggnet.pt'),
    ]
    
    ai_model = models.CharField(_('AI Models'),
                                  max_length=250,
                                  null=True,
                                  blank=True,
                                  choices=AIMODEL_CHOICES,
                                  default=AIMODEL_CHOICES[0],
                                  help_text="Selected yolo model will download. \
                                 Requires an active internet connection."
                                  )
    
    model_conf = models.DecimalField(_('Model confidence'),
                                     decimal_places=2,
                                     max_digits=4,
                                     null=True,
                                     blank=True)