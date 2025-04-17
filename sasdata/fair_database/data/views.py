import os

from django.contrib.auth.models import User
from django.shortcuts import get_object_or_404
from django.http import (
    HttpResponseBadRequest,
    HttpResponseForbidden,
    HttpResponse,
    Http404,
    FileResponse,
)
from rest_framework.response import Response
from rest_framework import status
from rest_framework.views import APIView

from sasdata.dataloader.loader import Loader
from data.serializers import (
    DataFileSerializer,
    DataSetSerializer,
    AccessManagementSerializer,
    SessionSerializer,
    PublishedStateSerializer,
    PublishedStateUpdateSerializer,
)
from data.models import DataFile, DataSet, PublishedState, Session
from data.forms import DataFileForm
from fair_database import permissions
from fair_database.permissions import DataPermission


class DataFileView(APIView):
    """
    View associated with the DataFile model.

    Functionality for viewing a list of files and uploading a new file.
    """

    # List of datafiles
    def get(self, request, version=None):
        if "username" in request.GET:
            search_user = get_object_or_404(User, username=request.GET["username"])
            data_list = {"user_data_ids": {}}
            private_data = DataFile.objects.filter(current_user=search_user)
            for x in private_data:
                if permissions.check_permissions(request, x):
                    data_list["user_data_ids"][x.id] = x.file_name
        else:
            public_data = DataFile.objects.all()
            data_list = {"public_data_ids": {}}
            for x in public_data:
                if permissions.check_permissions(request, x):
                    data_list["public_data_ids"][x.id] = x.file_name
        return Response(data_list)

    # Create a datafile
    def post(self, request, version=None):
        form = DataFileForm(request.data, request.FILES)
        if form.is_valid():
            form.save()
        db = DataFile.objects.get(pk=form.instance.pk)
        serializer = DataFileSerializer(
            db,
            data={
                "file_name": os.path.basename(form.instance.file.path),
                "current_user": None,
                "users": [],
            },
            context={"is_public": db.is_public},
        )
        if request.user.is_authenticated:
            serializer.initial_data["current_user"] = request.user.id

        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return_data = {
            "current_user": request.user.username,
            "authenticated": request.user.is_authenticated,
            "file_id": db.id,
            "file_alternative_name": serializer.data["file_name"],
            "is_public": serializer.data["is_public"],
        }
        return Response(return_data, status=status.HTTP_201_CREATED)

    # Create a datafile
    def put(self, request, version=None):
        return self.post(request, version)


class SingleDataFileView(APIView):
    """
    View associated with a single DataFile.

    Functionality for viewing, modifying, or deleting a DataFile.
    """

    # Load the contents of a datafile or download the file to a device
    def get(self, request, data_id, version=None):
        data = get_object_or_404(DataFile, id=data_id)
        if "download" in request.GET and request.GET["download"]:
            if not permissions.check_permissions(request, data):
                if not request.user.is_authenticated:
                    return HttpResponse("Must be authenticated to download", status=401)
                return HttpResponseForbidden("data is private")
            try:
                file = open(data.file.path, "rb")
            except Exception as e:
                return HttpResponseBadRequest(str(e))
            if file is None:
                raise Http404("File not found.")
            return FileResponse(file, as_attachment=True)
        else:
            loader = Loader()
            if not permissions.check_permissions(request, data):
                if not request.user.is_authenticated:
                    return HttpResponse("Must be authenticated to view", status=401)
                return HttpResponseForbidden(
                    "Data is either not public or wrong auth token"
                )
            data_list = loader.load(data.file.path)
            contents = [str(data) for data in data_list]
            return_data = {data.file_name: contents}
            return Response(return_data)

    # Modify a datafile
    def put(self, request, data_id, version=None):
        db = get_object_or_404(DataFile, id=data_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse("must be authenticated to modify", status=401)
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
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        return_data = {
            "current_user": request.user.username,
            "authenticated": request.user.is_authenticated,
            "file_id": db.id,
            "file_alternative_name": serializer.data["file_name"],
            "is_public": serializer.data["is_public"],
        }
        return Response(return_data)

    # Delete a datafile
    def delete(self, request, data_id, version=None):
        db = get_object_or_404(DataFile, id=data_id)
        if not permissions.is_owner(request, db):
            if not request.user.is_authenticated:
                return HttpResponse("Must be authenticated to delete", status=401)
            return HttpResponseForbidden("Must be the data owner to delete")
        db.delete()
        return Response(data={"success": True})


class DataFileUsersView(APIView):
    """
    View for the users that have access to a datafile.

    Functionality for accessing a list of users with access and granting or
    revoking access.
    """

    # View users with access to a datafile
    def get(self, request, data_id, version=None):
        db = get_object_or_404(DataFile, id=data_id)
        if not permissions.is_owner(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to manage access", status=401
                )
            return HttpResponseForbidden("Must be the data owner to manage access")
        response_data = {
            "file": db.pk,
            "file_name": db.file_name,
            "is_public": db.is_public,
            "users": [user.username for user in db.users.all()],
        }
        return Response(response_data)

    # Grant or revoke access to a datafile
    def put(self, request, data_id, version=None):
        db = get_object_or_404(DataFile, id=data_id)
        if not permissions.is_owner(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to manage access", status=401
                )
            return HttpResponseForbidden("Must be the data owner to manage access")
        serializer = AccessManagementSerializer(data=request.data)
        serializer.is_valid()
        user = get_object_or_404(User, username=serializer.data["username"])
        if serializer.data["access"]:
            db.users.add(user)
        else:
            db.users.remove(user)
        response_data = {
            "username": user.username,
            "file": db.pk,
            "file_name": db.file_name,
            "access": (serializer.data["access"] or user == db.current_user),
        }
        return Response(response_data)


class DataSetView(APIView):
    """
    View associated with the DataSet model.

    Functionality for viewing a list of datasets and creating a dataset.
    """

    permission_classes = [DataPermission]

    # get a list of accessible datasets
    def get(self, request, version=None):
        data_list = {"dataset_ids": {}}
        data = DataSet.objects.all()
        if "username" in request.GET:
            user = get_object_or_404(User, username=request.GET["username"])
            data = DataSet.objects.filter(current_user=user)
        for dataset in data:
            if permissions.check_permissions(request, dataset):
                data_list["dataset_ids"][dataset.id] = dataset.name
        return Response(data=data_list)

    # TODO: enable uploading files as part of dataset creation, not just associating dataset with existing files
    # create a dataset
    def post(self, request, version=None):
        # TODO: JSON deserialization probably
        # TODO: revisit request data format
        serializer = DataSetSerializer(data=request.data, context={"request": request})
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        db = serializer.instance
        response = {
            "dataset_id": db.id,
            "name": db.name,
            "authenticated": request.user.is_authenticated,
            "current_user": request.user.username,
            "is_public": db.is_public,
        }
        return Response(data=response, status=status.HTTP_201_CREATED)

    # create a dataset
    def put(self, request, version=None):
        return self.post(request, version)


class SingleDataSetView(APIView):
    """
    View associated with single datasets.

    Functionality for accessing a dataset in a format intended to be loaded
    into SasView, modifying a dataset, or deleting a dataset.
    """

    permission_classes = [DataPermission]

    # get a specific dataset
    def get(self, request, data_id, version=None):
        db = get_object_or_404(DataSet, id=data_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse("Must be authenticated to view dataset", status=401)
            return HttpResponseForbidden(
                "You do not have permission to view this dataset."
            )
        serializer = DataSetSerializer(db, context={"request": request})
        response_data = serializer.data
        if db.current_user:
            response_data["current_user"] = db.current_user.username
        return Response(response_data)

    # edit a specific dataset
    def put(self, request, data_id, version=None):
        db = get_object_or_404(DataSet, id=data_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to modify dataset", status=401
                )
            return HttpResponseForbidden("Cannot modify a dataset you do not own")
        serializer = DataSetSerializer(
            db, request.data, context={"request": request}, partial=True
        )
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        data = {"data_id": db.id, "name": db.name, "is_public": db.is_public}
        return Response(data)

    # delete a dataset
    def delete(self, request, data_id, version=None):
        db = get_object_or_404(DataSet, id=data_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to delete a dataset", status=401
                )
            return HttpResponseForbidden("Not authorized to delete")
        db.delete()
        return Response({"success": True})


class DataSetUsersView(APIView):
    """
    View for the users that have access to a dataset.

    Functionality for accessing a list of users with access and granting or
    revoking access.
    """

    permission_classes = [DataPermission]

    # get a list of users with access to dataset data_id
    def get(self, request, data_id, version=None):
        db = get_object_or_404(DataSet, id=data_id)
        if not permissions.is_owner(request, db):
            if not request.user.is_authenticated:
                return HttpResponse("Must be authenticated to view access", status=401)
            return HttpResponseForbidden("Must be the dataset owner to view access")
        response_data = {
            "data_id": db.id,
            "name": db.name,
            "is_public": db.is_public,
            "users": [user.username for user in db.users.all()],
        }
        return Response(response_data)

    # grant or revoke a user's access to dataset data_id
    def put(self, request, data_id, version=None):
        db = get_object_or_404(DataSet, id=data_id)
        if not permissions.is_owner(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to manage access", status=401
                )
            return HttpResponseForbidden("Must be the dataset owner to manage access")
        serializer = AccessManagementSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = get_object_or_404(User, username=serializer.data["username"])
        if serializer.data["access"]:
            db.users.add(user)
        else:
            db.users.remove(user)
        response_data = {
            "username": user.username,
            "data_id": db.id,
            "name": db.name,
            "access": serializer.data["access"],
        }
        return Response(response_data)


class SessionView(APIView):
    """
    View associated with the Session model.

    Functionality for viewing a list of sessions and for creating a session.
    """

    # View a list of accessible sessions
    def get(self, request, version=None):
        session_list = {"session_ids": {}}
        sessions = Session.objects.all()
        if "username" in request.GET:
            user = get_object_or_404(User, username=request.GET["username"])
            sessions = Session.objects.filter(current_user=user)
        for session in sessions:
            if permissions.check_permissions(request, session):
                session_list["session_ids"][session.id] = session.title
        return Response(data=session_list)

    # Create a session
    # TODO: revisit response data
    def post(self, request, version=None):
        serializer = SessionSerializer(data=request.data, context={"request": request})
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        db = serializer.instance
        response = {
            "session_id": db.id,
            "title": db.title,
            "authenticated": request.user.is_authenticated,
            "current_user": request.user.username,
            "is_public": db.is_public,
        }
        return Response(data=response, status=status.HTTP_201_CREATED)

    # Create a session
    def put(self, request, version=None):
        return self.post(request, version)


class SingleSessionView(APIView):
    """
    View associated with single sessions.

    Functionality for viewing, modifying, and deleting individual sessions.
    """

    # get a specific session
    def get(self, request, data_id, version=None):
        db = get_object_or_404(Session, id=data_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse("Must be authenticated to view session", status=401)
            return HttpResponseForbidden(
                "You do not have permission to view this session."
            )
        serializer = SessionSerializer(db)
        response_data = serializer.data
        if db.current_user:
            response_data["current_user"] = db.current_user.username
        return Response(response_data)

    # modify a session
    def put(self, request, data_id, version=None):
        db = get_object_or_404(Session, id=data_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to modify session", status=401
                )
            return HttpResponseForbidden("Cannot modify a session you do not own")
        serializer = SessionSerializer(
            db, request.data, context={"request": request}, partial=True
        )
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        data = {"session_id": db.id, "title": db.title, "is_public": db.is_public}
        return Response(data)

    # delete a session
    def delete(self, request, data_id, version=None):
        db = get_object_or_404(Session, id=data_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to delete a session", status=401
                )
            return HttpResponseForbidden("Not authorized to delete")
        db.delete()
        return Response({"success": True})


class SessionUsersView(APIView):
    """
    View for the users that have access to a session.

    Functionality for accessing a list of users with access and granting or
    revoking access.
    """

    # view the users that have access to a specific session
    def get(self, request, data_id, version=None):
        db = get_object_or_404(Session, id=data_id)
        if not permissions.is_owner(request, db):
            if not request.user.is_authenticated:
                return HttpResponse("Must be authenticated to view access", status=401)
            return HttpResponseForbidden("Must be the session owner to view access")
        response_data = {
            "session_id": db.id,
            "title": db.title,
            "is_public": db.is_public,
            "users": [user.username for user in db.users.all()],
        }
        return Response(response_data)

    # grant or revoke access to a session
    def put(self, request, data_id, version=None):
        db = get_object_or_404(Session, id=data_id)
        if not permissions.is_owner(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to manage access", status=401
                )
            return HttpResponseForbidden("Must be the dataset owner to manage access")
        serializer = AccessManagementSerializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        user = get_object_or_404(User, username=serializer.data["username"])
        if serializer.data["access"]:
            db.users.add(user)
            for dataset in db.datasets.all():
                dataset.users.add(user)
        else:
            db.users.remove(user)
            for dataset in db.datasets.all():
                dataset.users.remove(user)
        response_data = {
            "username": user.username,
            "session_id": db.id,
            "title": db.title,
            "access": serializer.data["access"],
        }
        return Response(response_data)


class PublishedStateView(APIView):
    """
    View associated with the PublishedState model.

    Functionality for viewing a list of session published states and for
    creating a published state.
    """

    # View a list of accessible sessions' published states
    def get(self, request, version=None):
        ps_list = {"published_state_ids": {}}
        published_states = PublishedState.objects.all()
        if "username" in request.GET:
            user = get_object_or_404(User, username=request.GET["username"])
            published_states = PublishedState.objects.filter(session__current_user=user)
        for ps in published_states:
            if permissions.check_permissions(request, ps.session):
                ps_list["published_state_ids"][ps.id] = {
                    "title": ps.session.title,
                    "published": ps.published,
                    "doi": ps.doi,
                }
        return Response(data=ps_list)

    # Create a published state for an existing session
    def post(self, request, version=None):
        serializer = PublishedStateSerializer(
            data=request.data, context={"request": request}
        )
        if serializer.is_valid(raise_exception=True):
            if not permissions.is_owner(request, serializer.validated_data["session"]):
                if not request.user.is_authenticated:
                    return HttpResponse(
                        "Must be authenticated to create a published state for a session",
                        status=401,
                    )
                return HttpResponseForbidden(
                    "Must be the session owner to create a published state for a session"
                )
            serializer.save()
        db = serializer.instance
        response = {
            "published_state_id": db.id,
            "session_id": db.session.id,
            "title": db.session.title,
            "doi": db.doi,
            "published": db.published,
            "current_user": request.user.username,
            "is_public": db.session.is_public,
        }
        return Response(data=response, status=status.HTTP_201_CREATED)

    # Create a published state for an existing session
    def put(self, request, version=None):
        return self.post(request, version)


class SinglePublishedStateView(APIView):
    """
    View associated with specific session published states.

    Functionality for viewing, modifying, and deleting individual published states.
    """

    # View a specific published state
    def get(self, request, ps_id, version=None):
        db = get_object_or_404(PublishedState, id=ps_id)
        if not permissions.check_permissions(request, db.session):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to view published state", status=401
                )
            return HttpResponseForbidden(
                "You do not have permission to view this published state."
            )
        serializer = PublishedStateSerializer(db)
        response_data = serializer.data
        if db.current_user:
            response_data["current_user"] = db.current_user.username
        return Response(response_data)

    # Modify a published state
    def put(self, request, ps_id, version=None):
        db = get_object_or_404(PublishedState, id=ps_id)
        if not permissions.check_permissions(request, db.session):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to modify published state", status=401
                )
            return HttpResponseForbidden(
                "Cannot modify a published state you do not own"
            )
        serializer = PublishedStateUpdateSerializer(
            db, request.data, context={"request": request}, partial=True
        )
        if serializer.is_valid(raise_exception=True):
            serializer.save()
        data = {
            "published_state_id": db.id,
            "session_id": db.session.id,
            "title": db.session.title,
            "published": db.published,
            "is_public": db.session.is_public,
        }
        return Response(data)

    # Delete a published state
    def delete(self, request, ps_id, version=None):
        db = get_object_or_404(PublishedState, id=ps_id)
        if not permissions.check_permissions(request, db):
            if not request.user.is_authenticated:
                return HttpResponse(
                    "Must be authenticated to delete a published state", status=401
                )
            return HttpResponseForbidden("Not authorized to delete")
        db.delete()
        return Response({"success": True})
