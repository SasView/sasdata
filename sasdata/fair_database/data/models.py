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


'''
class DataSet(Data):
    """Database model for a set of data and associated metadata."""

    # dataset name
    name = models.CharField(max_length=200)

    # associated files
    files = models.ManyToManyField(DataFile)

    # metadata
    # metadata = models.ForeignKey("MetaData", on_delete=models.CASCADE)

    # ordinate
    ordinate = models.JSONField()

    # abscissae
    abscissae = models.JSONField()

    # data contents
    data_contents = models.JSONField()

    # metadata
    raw_metadata = models.JSONField()


class MetaData:
    """Database model for scattering metadata"""

    # Associated data set
    # dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)


"""Database model for group of DataSets associated by a varying parameter."""


class OperationTree(Data):
    """Database model for tree of operations performed on a DataSet."""

    # Dataset the operation tree is performed on
    # dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)

    # operation

    # previous operation


class Session(Data):
    """Database model for a project save state."""

    # dataset
    # dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)

    # operation tree
    # operations = models.ForeignKey(OperationTree, on_delete=models.CASCADE)
'''
