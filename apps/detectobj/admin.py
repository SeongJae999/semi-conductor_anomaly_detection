from django.contrib import admin
from .models import InferencedImage

# Register your models here.

@admin.register(InferencedImage)
class InferencedImageAdmin(admin.ModelAdmin):
    list_display = ["orig_image", "inf_image_path",
                    "custom_model"]