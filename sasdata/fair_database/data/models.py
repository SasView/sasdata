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

    # TODO: Update when plan for this is finished.

    # dataset name
    name = models.CharField(max_length=200)

    # associated files
    files = models.ManyToManyField(DataFile)

    # metadata - maybe a foreign key?
    # TODO: when MetaData is finished, set blank/null false
    metadata = models.OneToOneField(
        "MetaData", blank=True, null=True, on_delete=models.CASCADE
    )

    # data contents - maybe ManyToManyField
    data_contents = models.ManyToManyField("Quantity", through="LabeledQuantity")

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


class LabeledQuantity(models.Model):
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)
    quantity = models.ForeignKey(Quantity, on_delete=models.CASCADE)
    label = models.CharField(max_length=20)


def empty_list():
    return []


def empty_dict():
    return {}


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

    # Dataset the operation tree is performed on
    dataset = models.ForeignKey(DataSet, on_delete=models.CASCADE)

    # operation
    operation = models.CharField(max_length=20, choices=OPERATION_CHOICES)

    # parameters
    parameters = models.JSONField(default=empty_dict)

    # previous operation
    parent_operation1 = models.ForeignKey(
        "self",
        blank=True,
        null=True,
        on_delete=models.CASCADE,
        related_name="child_operations1",
    )

    # optional second previous operation for binary operations
    parent_operation2 = models.ForeignKey(
        "self",
        blank=True,
        null=True,
        on_delete=models.CASCADE,
        related_name="child_operations2",
    )


'''
class Session(Data):
    """Database model for a project save state."""

    # dataset
    # dataset = models.ManyToManyField(DataSet)

    # operation tree
    # operations = models.ForeignKey(OperationTree, on_delete=models.CASCADE)

    published_state = models.ForeignKey("PublishedState", blank=True, null=True, on_delete=SET_NULL)

class PublishedState():
    """Database model for a project published state."""

    # published
    published = models.BooleanField(default=False)

    # doi
    doi = models.URLField()


'''
