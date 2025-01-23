from django.shortcuts import render

# Create your views here
from django.http import HttpResponse, HttpResponseBadRequest
from rest_framework.decorators import api_view
from rest_framework.response import Response

from .models import Data

@api_view(['GET'])
def list_data(request, username = None):
    if request.method == 'GET':
        if username:
            data_list = {"user_data_ids": {}}
            if username == request.user.username and request.user.is_authenticated:
                private_data = Data.objects.filter(current_user=request.user.id)
                for x in private_data:
                    data_list["user_data_ids"][x.id] = x.file_name
            else:
                return HttpResponseBadRequest("user is not logged in, or username is not same as current user")
        else:
            public_data = Data.objects.filter(is_public=True)
            data_list = {"public_data_ids": {}}
            for x in public_data:
                data_list["public_data_ids"][x.id] = x.file_name
        return Response(data_list)
    return HttpResponseBadRequest("not get method")

def data_info(request, db_id):
    return HttpResponse("This is going to allow viewing data file %s." % db_id)

def upload(request, db_id = None):
    return HttpResponse("This is going to allow data uploads.")

def download(request, data_id):
    return HttpResponse("This is going to allow downloads of data %s." % data_id)