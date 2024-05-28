from django.conf import settings
from django.contrib.auth.mixins import LoginRequiredMixin
from django.core.paginator import Paginator
from django.http import HttpResponse, HttpResponseRedirect
from django.shortcuts import render
from django.views.generic.detail import DetailView

from .forms import InferencedImageForm, AIModelForm
from .models import InferencedImage
from ast import literal_eval
from images.models import ImageFile
from keras import optimizers
from keras.models import model_from_json
from modelmanager.models import MLModel
from PIL import Image as I

import collections
import io
import os
import tensorflow as tf

from flask import render_template
from tensorflow.keras.applications import VGG16 
from tensorflow.keras.layers import Input
import glob
import json

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

    
    # 폴더안의 파일 삭제 함수
    def DeleteAllFiles(self, filePath):
        if os.path.exists(filePath):
            for file in os.scandir(filePath):
                os.remove(file.path)
            
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
        
        ai_model_name = self.request.POST.get("ai_model")
        
        json_file = open("C:/mango/vgg16.json", "r")
        loaded_json_model = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_json_model)
        loaded_model.load_weight("C:/mango/vgg16_model.h5")
        loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        
        model = VGG16(weight='imagenet')
        results = model(img, size=640)
        results_list = results.pandas().xyxy[0].to_json(orient="records")
        results_list = literal_eval(results_list)
        classes_list = [item["name"] for item in results_list]
        results_counter = collections.Counter(classes_list)

        results.render()
        
        data = results.pandas().xyxy[0][['name']].values.tolist()
        pf=[]
        
        if len(data) == 0:
            pf.append("PASS")    # data 리스트의 값이 0이면 양품으로 pass
        if len(data) != 0:
            pf.append("FAIL")  
        
        media_folder = settings.MEDIA_ROOT
        inferenced_img_dir = os.path.join(
            media_folder, "inferenced_image")
        if not os.path.exists(inferenced_img_dir):
            os.makedirs(inferenced_img_dir)
            
        for img in results.ims:
            img_base64 = I.fromarray(img)
            img_base64.save(
                f"{inferenced_img_dir}/{img_qs}", format="PNG")
        
        inf_img_qs, created = InferencedImage.objects.get_or_create(
            orig_image=img_qs,
            inf_image_path=f"{settings.MEDIA_URL}inferenced_image/{img_qs.name}",
        )
        inf_img_qs.detection_info = results_list
    
        inf_img_qs.ai_model = ai_model_name
        inf_img_qs.save()
        
        # set image is_inferenced to true
        img_qs.is_inferenced = True
        img_qs.save()
        # Ready for rendering next image on same html page.
        imgset = img_qs.image_set
        images_qs = imgset.images.all()
        

        resultlist=[]
        for file in file:
            
            root = "static/aft"
            if not os.path.isdir(root):      #파일명 리스트로 저장
                return "Error : not found!"
            files = []
            for file in glob.glob("{}/*.*".format(root)):
                fname = file.split(os.sep)[-1]
                files.append(fname)
            print("파일스 :",files)
            
            if len(files)>0:
                firstimage = "static/aft/"+files[0]
            else: pass

            datanum = len(pf)
            rate = round(pf.count('PASS') / len(pf), 3)
            correct = pf.count('PASS')
            
        # For pagination POST request
        context = {}
        self.get_pagination(context, images_qs)
        context["files"] = files
        context["resultlist"] = resultlist
        context["pf"] = pf
        context["datanum"] = datanum
        context["rate"] = rate
        context["correct"] = correct
        context["firstimage"] = firstimage
        context["enumerate"] = enumerate
        context["len"] = len
        context["results_list"] = results_list
        context["img_qs"] = False
        context["inferenced_img_dir"] = f"{settings.MEDIA_URL}inferenced_image/{img_qs}"
        context["form1"] = AIModelForm()
        context["form2"] = InferencedImageForm()
        return render(self.request, self.template_name, context)