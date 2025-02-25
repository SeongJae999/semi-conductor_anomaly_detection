from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import Paginator
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.views.generic.detail import DetailView

from .forms import InferencedImageForm, AIModelForm
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from .models import InferencedImage
from ast import literal_eval
from images.models import ImageFile
from keras import optimizers
from keras.models import model_from_json
from modelmanager.models import MLModel
from PIL import Image as I

import collections
import io
import numpy as np
import os
import pandas as pd
import tensorflow as tf

from flask import render_template
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.layers import Input
import glob
import json
import yolov5

# Create your views here.
class InferencedImageDetectionView(LoginRequiredMixin, DetailView):
    model = ImageFile
    template_name = "detectobj/select_inference_image.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        img_qs = self.get_object()
        imgset = img_qs.image_set
        images_qs = imgset.images.all()

        # For pagination GET request
        self.get_pagination(context, images_qs)
        
        if is_inf_img := InferencedImage.objects.filter(
            orig_image=img_qs
        ).exists():
            inf_img_qs = InferencedImage.objects.get(orig_image=img_qs)
            context['inf_img_qs'] = inf_img_qs
        
        context["img_qs"] = img_qs
        context["form1"] = AIModelForm()
        context["form2"] = InferencedImageForm()
        return context

    def get_pagination(self, context, images_qs):
        paginator = Paginator(
            images_qs, settings.PAGINATE_DETECTION_IMAGES_NUM)
        page_number = self.request.GET.get('page')
        page_obj = paginator.get_page(page_number)
        context["is_paginated"] = (
            images_qs.count() > settings.PAGINATE_DETECTION_IMAGES_NUM
        )
        context["page_obj"] = page_obj
                    
    def post(self, request, *args, **kwargs):
        img_qs = self.get_object()
        img_bytes = img_qs.image.read()
        img = I.open(io.BytesIO(img_bytes))

        # Get form data
        custom_model_id = self.request.POST.get("custom_model")
        yolo_model_name = self.request.POST.get("ai_model")

        # Yolov5 dirs
        yolo_weightsdir = settings.YOLOV5_WEIGTHS_DIR

        # Whether user selected a custom model for the detection task
        # An offline model will be used for detection provided user has
        # uploaded this model.
        if custom_model_id:
            detection_model = MLModel.objects.get(id=custom_model_id)
            model = yolov5.load(detection_model.pth_filepath)
            # print(model.names)

        # Whether user selected a yolo model for the detection task
        # Selected yolov5 model will be downloaded, and ready for object
        # detection task. Yolov5 Api's will start working.
        elif yolo_model_name:
            model = yolov5.load(os.path.join(yolo_weightsdir, yolo_model_name))

        # classnames = model.names  (display classes in the model)
        results_list = []
        results = model(img, size=(27,27))
        results_list = results.pandas().xyxy[0].to_json(orient="records")
        results_list = literal_eval(results_list)
        classes_list = [item["name"] for item in results_list]
        results_counter = collections.Counter(classes_list)
        # if results_list == []:
        #     messages.warning(
        #         request, f'Model "{detection_model.name}" unable to predict. Try with another model.')
        # else:
        results.render()
        
        
        media_folder = settings.MEDIA_ROOT
        inferenced_img_dir = os.path.join(
            media_folder, "inferenced_image")
        if not os.path.exists(inferenced_img_dir):
            os.makedirs(inferenced_img_dir)

        # print(dir(results))
        for img in results.ims:
            img_base64 = I.fromarray(img)
            img_base64.save(
                f"{inferenced_img_dir}/{img_qs}", format="JPEG")

        # Create/update the inferencedImage instance
        inf_img_qs, created = InferencedImage.objects.get_or_create(
            orig_image=img_qs,
            inf_image_path=f"{settings.MEDIA_URL}inferenced_image/{img_qs.name}",
        )
        inf_img_qs.detection_info = results_list
        if custom_model_id:
            inf_img_qs.custom_model = detection_model
        elif yolo_model_name:
            inf_img_qs.yolo_model = yolo_model_name
        inf_img_qs.save()
        
        results_list.append(json.dumps(dict(results_list)))
        # set image is_inferenced to true
        img_qs.is_inferenced = True
        img_qs.save()
        # Ready for rendering next image on same html page.
        imgset = img_qs.image_set
        images_qs = imgset.images.all()

        # For pagination POST request
        context = {}
        self.get_pagination(context, images_qs)

        context["img_qs"] = img_qs
        context["inferenced_img_dir"] = f"{settings.MEDIA_URL}inferenced_image/{img_qs}"
        context["results_list"] = results_list
        context["results_counter"] = results_counter
        context["form1"] = AIModelForm()
        context["form2"] = InferencedImageForm()
        return render(self.request, self.template_name, context)