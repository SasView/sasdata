from django.urls import path

from . import views

urlpatterns = [
    path("file/", views.DataFileView.as_view(), name="view and create files"),
    path(
        "file/<int:data_id>/",
        views.SingleDataFileView.as_view(),
        name="view, download, modify, delete files",
    ),
    path(
        "file/<int:data_id>/users/",
        views.DataFileUsersView.as_view(),
        name="manage access to files",
    ),
    path("set/", views.DataSetView.as_view(), name="view and create datasets"),
    path(
        "set/<int:data_id>/",
        views.SingleDataSetView.as_view(),
        name="load, modify, delete datasets",
    ),
    path(
        "set/<int:data_id>/users/",
        views.DataSetUsersView.as_view(),
        name="manage access to datasets",
    ),
    path("session/", views.SessionView.as_view(), name="view and create sessions"),
    path(
        "session/<int:data_id>/",
        views.SingleSessionView.as_view(),
        name="load, modify, delete sessions",
    ),
    path(
        "session/<int:data_id>/users/",
        views.SessionUsersView.as_view(),
        name="manage access to sessions",
    ),
]
