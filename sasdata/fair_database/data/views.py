from django.shortcuts import render

# Create your views here
from django.http import HttpResponse

def list_data(request):
    return HttpResponse("Hello World! This is going to display data later.")