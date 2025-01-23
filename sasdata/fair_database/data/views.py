from django.shortcuts import render

# Create your views here
from django.http import HttpResponse

def list_data(request, username = None):
    return HttpResponse("Hello World! This is going to display data later.")

def data_info(request, db_id):
    return HttpResponse("This is going to allow viewing data file %s." % db_id)

def upload(request, db_id = None):
    return HttpResponse("This is going to allow data uploads.")

def download(request, data_id):
    return HttpResponse("This is going to allow downloads of data %s." % data_id)