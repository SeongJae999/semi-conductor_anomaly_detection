from django.urls import path
from . import views

app_name = "detectobj"

urlpatterns = [
    path("<int:pk>/selected_image/",
         views.InferencedImageDetectionView.as_view(),
         name="detection_image_detail_url"
         ),
    path("<int:pk>/image_result/",
         views.InferencedImageResultView.as_view(),
         name="detection_image_result"
         ),
]