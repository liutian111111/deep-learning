# -*- coding: utf-8 -*-

#from django.http import HttpResponse
from django.shortcuts import render
import os
import sys
sys.path.append('/home/gpu/tensorflow/tensorflow/ai-cap/')
from model_predict import tmain
from mv_file import mv_file

def hello(request):
    context = {}
    mv_file()
    photo_url='/staticfiles/'+os.listdir('/home/gpu/tensorflow/tensorflow/ai-cap/show_report/aicap/template/static')[0]
    pre_result=''
    pre_result=tmain()
    context['photourl']=photo_url
    context['number']=pre_result
    return render(request, 'report.html', context)

