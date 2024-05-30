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

    #wafer map dimension initialize
    def find_dim(self, x):
        dim0=np.size(x,axis=0)
        dim1=np.size(x,axis=1)
        return dim0,dim1

    def test_data_preprocessing(self):
        df = pd.read_pickle("C:\mango\LSWMD.pkl")
        df = df.drop(['waferIndex'], axis = 1)
        df['waferMapDim']=df.waferMap.apply(self.find_dim)
        df['failureNum']=df.failureType
        df['trainTestNum']=df.trianTestLabel
        
        mapping_type={'Center':0,'Donut':1,'Edge-Loc':2,'Edge-Ring':3,'Loc':4,'Random':5,'Scratch':6,'Near-full':7,'none':8}
        mapping_traintest={'Training':0,'Test':1}
        df = df.replace({'failureNum':mapping_type, 'trainTestNum':mapping_traintest})
        
        x_test = x_test.reshape((-1, 27, 27, 1))
        
        sub_df = df.loc[df['waferMapDim'] == (25, 27)]

        sw = np.ones((1, 25, 27))
        label = list()

        for i in range(len(sub_df)):
            if len(sub_df.iloc[i,:]['failureType']) == 0:
                continue
            sw = np.concatenate((sw, sub_df.iloc[i,:]['waferMap'].reshape(1, 25, 27)))
            label.append(sub_df.iloc[i,:]['failureType'][0][0])
            
        x1 = sw[1:]
        # 27 x 27 사이즈로 맞추기 위한 padding 처리
        x = np.pad(x1, ((0,0),(1,1),(0,0)), mode='constant', constant_values = (0,0))
        y = np.array(label).reshape((-1,1))    
        
        sub_df1 = df.loc[df['waferMapDim'] == (26, 26)]

        sw1 = np.ones((1, 26, 26))
        label1 = list()

        for i in range(len(sub_df1)):
            if len(sub_df1.iloc[i,:]['failureType']) == 0:
                continue
            sw1 = np.concatenate((sw1, sub_df1.iloc[i,:]['waferMap'].reshape(1, 26, 26)))
            label1.append(sub_df1.iloc[i,:]['failureType'][0][0])
            
        x2 = sw1[1:]
        x3 = np.pad(x2, ((0,0),(1,0),(1,0)), mode='constant', constant_values = (0,0))
        y2 = np.array(label1).reshape((-1,1))
        
        sub_df2 = df.loc[df['waferMapDim'] == (27, 25)]

        sw2 = np.ones((1, 27, 25))
        label2= list()

        for i in range(len(sub_df2)):
            if len(sub_df2.iloc[i,:]['failureType']) == 0:
                continue
            sw2 = np.concatenate((sw2, sub_df2.iloc[i,:]['waferMap'].reshape(1, 27, 25)))
            label2.append(sub_df2.iloc[i,:]['failureType'][0][0])

        x4 = sw2[1:]
        x5 = np.pad(x4, ((0,0),(0,0),(1,1)), mode='constant', constant_values = (0,0))
        y3 = np.array(label2).reshape((-1,1))
        
        x3 = np.concatenate((x3, x5[0:]))
        y2 = np.concatenate((y2, y3))
        
        x = np.concatenate((x, x3[0:]))
        y = np.concatenate((y, y2))
        
        faulty_case = np.unique(y)
        
        # '감지되지 않는' none 불량 라벨 제거
        none_idx = np.where(y=='none')[0][np.random.choice(len(np.where(y=='none')[0]), size=25000, replace=False)]
        new_x = np.delete(x, none_idx, axis=0)
        new_y = np.delete(y, none_idx, axis=0)
        
        #train과 test 데이터를 8:2 비율로 맞춤
        x_train_temp, x_test, y_train_temp, y_test = train_test_split(new_x, new_y, test_size=0.2, random_state=789)    
        
        #x_test : 채널 추가
        x_test = x_test.reshape((-1, 27, 27, 1))
        
        # One-hot-Encoding 
        nx_test = np.zeros((len(x_test), 27, 27, 3))

        for w in range(len(x_test)):
            for i in range(27):
                for j in range(27):
                    nx_test[w, i, j, int(x_test[w, i, j])] = 1
                    
        #y_test : 문자열 데이터를 정수형으로 형변환
        for i, l in enumerate(faulty_case):
            y_test[y_test==l] = i
            
        # one-hot-encoding - 10진 정수 형식을 2진 바이너리 형식으로 변경
        y_test = to_categorical(y_test)

        for w in range(len(x_test)):
            for i in range(27):
                for j in range(27):
                    nx_test[w, i, j, int(x_test[w, i, j])] = 1
            
        return nx_test, y_test, faulty_case
                    
    def post(self, request, *args, **kwargs):
        img_qs = self.get_object()
        img_bytes = img_qs.image.read()
        img = I.open(io.BytesIO(img_bytes))
        
        ai_model_name = self.request.POST.get("ai_model")
        
        json_file = open("vgg16.json", "r")
        loaded_json_model = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_json_model)
        loaded_model.load_weights("vgg16_model.h5")
        loaded_model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])
        
        x_data, y_data, faulty_case = self.test_data_preprocessing()
        
        vy_pred = loaded_model.predict(x_data)
        
        vy_test_decode = np.ones(len(y_data))
        vy_pred_decode = np.ones(len(vy_pred))

        for i in range(len(vy_pred)):
            vy_pred_decode[i] = np.argmax(vy_pred[i])
            vy_test_decode[i] = np.argmax(y_data[i])
        
        inf_img_qs, created = InferencedImage.objects.get_or_create(
            orig_image=img_qs,
            inf_image_path=f"{settings.MEDIA_URL}inferenced_image/{img_qs.name}",
        )
    
        inf_img_qs.ai_model = ai_model_name
        inf_img_qs.save()
        
        # set image is_inferenced to true
        img_qs.is_inferenced = True
        img_qs.save()
        # Ready for rendering next image on same html page.
        imgset = img_qs.image_set
        images_qs = imgset.images.all()
            
        # For pagination POST request
        context = {}
        self.get_pagination(context, images_qs)
        context["enumerate"] = enumerate
        context["len"] = len
        context["img_qs"] = False
        context["inferenced_img_dir"] = f"{settings.MEDIA_URL}inferenced_image/{img_qs}"
        context["form1"] = AIModelForm()
        context["form2"] = InferencedImageForm()
        return render(self.request, self.template_name, context)