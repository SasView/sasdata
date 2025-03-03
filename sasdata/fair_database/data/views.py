import os

from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from django.http import (
    HttpResponseBadRequest,
    HttpResponseForbidden,
    Http404,
    FileResponse,
)
from rest_framework.decorators import api_view
from rest_framework.response import Response

from sasdata.dataloader.loader import Loader
from data.serializers import DataFileSerializer, AccessManagementSerializer
from data.models import DataFile
from data.forms import DataFileForm
from fair_database import permissions


@api_view(["GET"])
def list_data(request, username=None, version=None):
    if request.method == "GET":
        if username:
            search_user = get_object_or_404(User, username=username)
            data_list = {"user_data_ids": {}}
            private_data = DataFile.objects.filter(current_user=search_user)
            for x in private_data:
                if permissions.check_permissions(request, x):
                    data_list["user_data_ids"][x.id] = x.file_name
        else:
            public_data = DataFile.objects.filter(is_public=True)
            data_list = {"public_data_ids": {}}
            for x in public_data:
                if not permissions.check_permissions(request, x):
                    return HttpResponseForbidden()
                data_list["public_data_ids"][x.id] = x.file_name
        return Response(data_list)
    return HttpResponseBadRequest("not get method")


@api_view(["GET"])
def data_info(request, db_id, version=None):
    if request.method == "GET":
        loader = Loader()
        data_db = get_object_or_404(DataFile, id=db_id)
        if not permissions.check_permissions(request, data_db):
            return HttpResponseForbidden(
                "Data is either not public or wrong auth token"
            )
        data_list = loader.load(data_db.file.path)
        contents = [str(data) for data in data_list]
        return_data = {data_db.file_name: contents}
        return Response(return_data)
    return HttpResponseBadRequest()


@api_view(["POST", "PUT"])
def upload(request, data_id=None, version=None):
    # saves file
    if request.method in ["POST", "PUT"] and data_id is None:
        form = DataFileForm(request.data, request.FILES)
        if form.is_valid():
            form.save()
        db = DataFile.objects.get(pk=form.instance.pk)

        if request.user.is_authenticated:
            serializer = DataFileSerializer(
                db,
                data={
                    "file_name": os.path.basename(form.instance.file.path),
                    "current_user": request.user.id,
                    "users": [request.user.id],
                },
                context={"is_public": db.is_public},
            )
        else:
            serializer = DataFileSerializer(
                db,
                data={
                    "file_name": os.path.basename(form.instance.file.path),
                    "current_user": None,
                    "users": [],
                },
                context={"is_public": db.is_public},
            )

    # updates file
    elif request.method == "PUT":
        db = get_object_or_404(DataFile, id=data_id)
        if not permissions.check_permissions(request, db):
            return HttpResponseForbidden("must be the data owner to modify")
        form = DataFileForm(request.data, request.FILES, instance=db)
        if form.is_valid():
            form.save()
        serializer = DataFileSerializer(
            db,
            data={
                "file_name": os.path.basename(form.instance.file.path),
                "current_user": request.user.id,
            },
            context={"is_public": db.is_public},
            partial=True,
        )
    else:
        return HttpResponseBadRequest()

    if serializer.is_valid(raise_exception=True):
        serializer.save()
        # TODO get warnings/errors later
    return_data = {
        "current_user": request.user.username,
        "authenticated": request.user.is_authenticated,
        "file_id": db.id,
        "file_alternative_name": serializer.data["file_name"],
        "is_public": serializer.data["is_public"],
    }
    return Response(return_data)


# view or control who has access to a file
@api_view(["GET", "PUT"])
def manage_access(request, data_id, version=None):
    db = get_object_or_404(DataFile, id=data_id)
    if not permissions.is_owner(request, db):
        return HttpResponseForbidden("Must be the data owner to manage access")
    if request.method == "GET":
        response_data = {
            "file": db.pk,
            "file_name": db.file_name,
            "users": [user.username for user in db.users],
        }
        return Response(response_data)
    elif request.method == "PUT":
        serializer = AccessManagementSerializer(data=request.data)
        serializer.is_valid()
        user = get_object_or_404(User, username=serializer.data["username"])
        if serializer.data["access"]:
            db.users.add(user)
        else:
            db.users.remove(user)
        response_data = {
            "user": user.username,
            "file": db.pk,
            "file_name": db.file_name,
            "access": serializer.data["access"],
        }
        return Response(response_data)
    return HttpResponseBadRequest()


# downloads a file
@api_view(["GET"])
def download(request, data_id, version=None):
    if request.method == "GET":
        data = get_object_or_404(DataFile, id=data_id)
        if not permissions.check_permissions(request, data):
            return HttpResponseForbidden("data is private")
        # TODO add issues later
        try:
            file = open(data.file.path, "rb")
        except Exception as e:
            return HttpResponseBadRequest(str(e))
        if file is None:
            raise Http404("File not found.")
        return FileResponse(file, as_attachment=True)
    return HttpResponseBadRequest()
