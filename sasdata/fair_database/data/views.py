from django.shortcuts import render
from django.shortcuts import get_object_or_404

# Create your views here
from django.http import HttpResponse, HttpResponseBadRequest
from rest_framework.decorators import api_view
from rest_framework.response import Response

from sasdata.dataloader.loader import Loader
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

@api_view(['GET'])
def data_info(request, db_id):
    if request.method == 'GET':
        loader = Loader()
        data_db = get_object_or_404(Data, id=db_id)
        if data_db.is_public:
            data_list = loader.load(data_db.file.path)
            contents = [str(data) for data in data_list]
            return_data = {data_db.file_name: contents}
        # rewrite with "user.is_authenticated"
        elif (data_db.current_user == request.user) and request.user.is_authenticated:
            data_list = loader.load(data_db.file.path)
            contents = [str(data) for data in data_list]
            return_data = {data_db.file_name: contents}
        else:
            return HttpResponseBadRequest("Database is either not public or wrong auth token")
        return Response(return_data)
    return HttpResponseBadRequest()

def upload(request, db_id = None):
    return HttpResponse("This is going to allow data uploads.")

def download(request, data_id):
    return HttpResponse("This is going to allow downloads of data %s." % data_id)