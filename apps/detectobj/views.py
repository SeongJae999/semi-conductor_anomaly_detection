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
        self.DeleteAllFiles('C:/Users/admin/project3/project/project3/project3/static/aft')  #파일 업로드 후 새로 검사 시작할 때마다 폴더 내 파일 삭제
        self.DeleteAllFiles('C:/Users/admin/project3/project/project3/project3/static/bef')  # 경로설정필요
        
        # 다중파일 업로드
        if "file" not in request.FILES:
            return HttpResponseRedirect(request.get_full_path())
        file = request.FILES.getlist("file")
        if not file:
            return HttpResponse("Retry plz")
        
        model = VGG16(weight='imagenet')
        resultlist=[]
        pf=[]
        for file in file:
            filename = file.filename.rsplit("/")[0]     #파일경로에서 파일명만 추출
            print("진행 중 파일 :", filename)

            img_bytes = file.read()
            img = I.open(io.BytesIO(img_bytes))
            # print(img)
            img.convert("RGB").save(f"static/bef/{filename}", format="PNG")
            print('원본 저장')

            results = model(img, size=640)
            results_list = results.pandas().xyxy[0].to_json(orient="records")
            results_list = literal_eval(results_list)
            classes_list = [item["name"] for item in results_list]
            result_counter = collections.Counter(classes_list)
            
            results.render()  # results.imgs에 바운딩박스와 라벨 처리

            for img in results.ims:
                img_base64 = I.fromarray(img)
                img_base64.convert("RGB").save(f"static/aft/{filename}", format="PNG")
                print('디텍트 저장')

            resultlist.append(json.dumps(dict(result_counter)))

            data = results.pandas().xyxy[0][['name']].values.tolist()   # results.imgs의 name값만 가져오기
            print("데이터:",data)

            if len(data) == 0:
                pf.append("PASS")    # data 리스트의 값이 0이면 양품으로 pass
            if len(data) != 0:
                pf.append("FAIL")    # data 리스트의 값이 0이 아니면 불량으로 fail
            
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
            
            print(resultlist)

        # For pagination POST request
        context = {}
        

        
        
        context["results_list"] = results_list
        
        context["form1"] = AIModelForm()
        context["form2"] = InferencedImageForm()
        return render(self.request, self.template_name, context)