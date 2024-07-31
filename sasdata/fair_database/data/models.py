from django.db import models
from django.contrib.auth.models import User
from django.core.files.storage import FileSystemStorage


# method for empty list default value
def empty_list():
    return []


# method for empty dictionary default value
def empty_dict():
    return {}


class Data(models.Model):
    """Base model for data with access-related information."""

    #  owner of the data
    current_user = models.ForeignKey(
        User, blank=True, null=True, on_delete=models.CASCADE, related_name="+"
    )

    # users that have been granted view access to the data
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

    # session the dataset is a part of, if any
    session = models.ForeignKey(
        "Session",
        on_delete=models.CASCADE,
        related_name="datasets",
        blank=True,
        null=True,
    )

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

    # label, e.g. Q or I(Q)
    label = models.CharField(max_length=50)

    # data set the quantity is a part of
    dataset = models.ForeignKey(
        DataSet, on_delete=models.CASCADE, related_name="data_contents"
    )


class ReferenceQuantity(models.Model):
    """
    Database models for quantities referenced by variables in an OperationTree.

    Corresponds to the references dictionary in the QuantityHistory class in
    sasdata/quantity.py. ReferenceQuantities should be essentially the same as
    Quantities but with no operations performed on them and therefore no
    OperationTree.
    """

    # data value
    value = models.JSONField()

    # variance of the data
    variance = models.JSONField()

    # units
    units = models.CharField(max_length=200)

    # hash value
    hash = models.IntegerField()

    # Quantity whose OperationTree this is a reference for
    derived_quantity = models.ForeignKey(
        Quantity,
        related_name="references",
        on_delete=models.CASCADE,
    )


# TODO: update based on changes in sasdata/metadata.py
class MetaData(models.Model):
    """Database model for scattering metadata"""

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

    # possible operations
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

    # label (a or b) if the operation is a parameter of a child operation
    # maintains ordering of binary operation parameters
    label = models.CharField(max_length=10, blank=True, null=True)

    # operation this operation is a parameter for, if any
    child_operation = models.ForeignKey(
        "self",
        on_delete=models.CASCADE,
        related_name="parent_operations",
        blank=True,
        null=True,
    )

    # quantity the operation produces
    # only set for base of tree (the quantity's most recent operation)
    quantity = models.OneToOneField(
        Quantity,
        on_delete=models.CASCADE,
        blank=True,
        null=True,
        related_name="operation_tree",
    )


class Session(Data):
    """Database model for a project save state."""

    # title
    title = models.CharField(max_length=200)


class PublishedState(models.Model):
    """Database model for a project published state."""

    # published
    published = models.BooleanField(default=False)

    # TODO: update doi as needed when DOI generation is implemented
    # doi
    doi = models.URLField()

    # session
    session = models.OneToOneField(
        Session, on_delete=models.CASCADE, related_name="published_state"
    )
