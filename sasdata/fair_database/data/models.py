from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage


class Data(models.Model):
    """Base model for data."""

    #  owner of the data
    current_user = models.ForeignKey(
        User, blank=True, null=True, on_delete=models.CASCADE, related_name="+"
    )

    users = models.ManyToManyField(User, blank=True, related_name="+")

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


class DataSet(Data):
    """Database model for a set of data and associated metadata."""

    # dataset name
    name = models.CharField(max_length=200)

    # associated files
    files = models.ManyToManyField(DataFile)

    # TODO: update based on SasData class in data.py
    # type of dataset
    # dataset_type = models.JSONField()


class Quantity(models.Model):
    """Database model for data quantities such as the ordinate and abscissae."""

    # data value
    value = models.JSONField()

    # variance of the data
    variance = models.JSONField()

    # units
    units = models.CharField(max_length=200)

    # hash value
    hash = models.IntegerField()

    # TODO: add field to store references portion of QuantityHistory
    # operation history of the quantity - operation_tree from QuantityHistory
    operation_tree = models.OneToOneField(
        "OperationTree", blank=True, null=True, on_delete=models.SET_NULL
    )

    label = models.CharField(max_length=50)

    dataset = models.ForeignKey(
        DataSet, on_delete=models.CASCADE, related_name="data_contents"
    )


def empty_list():
    return []


def empty_dict():
    return {}


class MetaData(models.Model):
    """Database model for scattering metadata"""

    # TODO: update based on changes in sasdata/metadata.py
    # title
    title = models.CharField(max_length=500, default="Title")

    # run
    run = models.JSONField(default=empty_list)

    # definition
    definition = models.TextField(blank=True, null=True)

    # instrument
    instrument = models.JSONField(blank=True, null=True)

    # process
    process = models.JSONField(default=empty_list)

    # sample
    sample = models.JSONField(blank=True, null=True)

    # associated dataset
    dataset = models.OneToOneField(
        DataSet, on_delete=models.CASCADE, related_name="metadata"
    )


class OperationTree(models.Model):
    """Database model for tree of operations performed on a DataSet."""

    OPERATION_CHOICES = {
        "zero": "0 [Add.Id.]",
        "one": "1 [Mul.Id.]",
        "constant": "Constant",
        "variable": "Variable",
        "neg": "Neg",
        "reciprocal": "Inv",
        "add": "Add",
        "sub": "Sub",
        "mul": "Mul",
        "div": "Div",
        "pow": "Pow",
        "transpose": "Transpose",
        "dot": "Dot",
        "matmul": "MatMul",
        "tensor_product": "TensorProduct",
    }

    # operation
    operation = models.CharField(max_length=20, choices=OPERATION_CHOICES)

    # parameters
    parameters = models.JSONField(default=empty_dict)

    # previous operation
    parent_operation1 = models.ForeignKey(
        "self",
        blank=True,
        null=True,
        on_delete=models.PROTECT,
        related_name="child_operations1",
    )

    # optional second previous operation for binary operations
    parent_operation2 = models.ForeignKey(
        "self",
        blank=True,
        null=True,
        on_delete=models.PROTECT,
        related_name="child_operations2",
    )


class Session(Data):
    """Database model for a project save state."""

    # title
    title = models.CharField(max_length=200)

    # dataset
    dataset = models.ManyToManyField(DataSet)

    # publishing state of the session
    published_state = models.OneToOneField(
        "PublishedState", blank=True, null=True, on_delete=models.SET_NULL
    )


class PublishedState(models.Model):
    """Database model for a project published state."""

    # published
    published = models.BooleanField(default=False)

    # doi
    doi = models.URLField()
