from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage


class Data(models.Model):
    """Base model for data."""

    # username
    current_user = models.ForeignKey(
        User, blank=True, null=True, on_delete=models.CASCADE
    )

    # is the data public?
    is_public = models.BooleanField(
        default=False, help_text="opt in to make your data public"
    )

    class Meta:
        abstract = True


class DataFile(Data):
    """Database model for file contents."""

    # file name
    file_name = models.CharField(
        max_length=200, default=None, blank=True, null=True, help_text="File name"
    )

    # imported data
    # user can either import a file path or actual file
    file = models.FileField(
        blank=False,
        default=None,
        help_text="This is a file",
        upload_to="uploaded_files",
        storage=FileSystemStorage(),
    )


"""Database model for a set of data and associated metadata."""

"""Database model for group of DataSets associated by a varying parameter."""

"""Database model for tree of operations performed on a DataSet."""

"""Database model for a project save state."""
