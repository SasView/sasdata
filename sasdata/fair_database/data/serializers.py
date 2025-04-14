from rest_framework import serializers

from data import models


# TODO: more custom validation, particularly for specific nested dictionary structures


class DataFileSerializer(serializers.ModelSerializer):
    """Serialization, deserialization, and validation for the DataFile model."""

    class Meta:
        model = models.DataFile
        fields = "__all__"

    # TODO: check partial updates
    # Check that private data has an owner
    def validate(self, data):
        if not self.context["is_public"] and not data["current_user"]:
            raise serializers.ValidationError("private data must have an owner")
        return data


class AccessManagementSerializer(serializers.Serializer):
    """
    Serialization, deserialization, and validation for granting and revoking
    access to instances of any exposed model.
    """

    # The username of a user
    username = serializers.CharField(max_length=200, required=False)
    # Whether that user has access
    access = serializers.BooleanField()


class MetaDataSerializer(serializers.ModelSerializer):
    """Serialization, deserialization, and validation for the MetaData model."""

    dataset = serializers.PrimaryKeyRelatedField(
        queryset=models.DataSet, required=False, allow_null=True
    )

    class Meta:
        model = models.MetaData
        fields = "__all__"

    # Serialize an entry in MetaData
    def to_representation(self, instance):
        data = super().to_representation(instance)
        if "dataset" in data:
            data.pop("dataset")
        return data

    # Create an entry in MetaData
    def create(self, validated_data):
        dataset = models.DataSet.objects.get(id=validated_data.pop("dataset"))
        return models.MetaData.objects.create(dataset=dataset, **validated_data)


class OperationTreeSerializer(serializers.ModelSerializer):
    """Serialization, deserialization, and validation for the OperationTree model."""

    quantity = serializers.PrimaryKeyRelatedField(
        queryset=models.Quantity, required=False, allow_null=True
    )
    child_operation = serializers.PrimaryKeyRelatedField(
        queryset=models.OperationTree, required=False, allow_null=True
    )
    label = serializers.CharField(max_length=10, required=False)

    class Meta:
        model = models.OperationTree
        fields = ["operation", "parameters", "quantity", "label", "child_operation"]

    # Validate parent operations
    def validate_parameters(self, value):
        if "a" in value:
            serializer = OperationTreeSerializer(data=value["a"])
            serializer.is_valid(raise_exception=True)
        if "b" in value:
            serializer = OperationTreeSerializer(data=value["b"])
            serializer.is_valid(raise_exception=True)
        return value

    # Check that the operation has the correct parameters
    def validate(self, data):
        expected_parameters = {
            "zero": [],
            "one": [],
            "constant": ["value"],
            "variable": ["hash_value", "name"],
            "neg": ["a"],
            "reciprocal": ["a"],
            "add": ["a", "b"],
            "sub": ["a", "b"],
            "mul": ["a", "b"],
            "div": ["a", "b"],
            "pow": ["a", "power"],
            "transpose": ["a", "axes"],
            "dot": ["a", "b"],
            "matmul": ["a", "b"],
            "tensor_product": ["a", "b", "a_index", "b_index"],
        }

        for parameter in expected_parameters[data["operation"]]:
            if parameter not in data["parameters"]:
                raise serializers.ValidationError(
                    data["operation"] + " requires parameter " + parameter
                )

        return data

    # Serialize an OperationTree instance
    def to_representation(self, instance):
        data = {"operation": instance.operation, "parameters": instance.parameters}
        for parent_operation in instance.parent_operations.all():
            data["parameters"][parent_operation.label] = self.to_representation(
                parent_operation
            )
        return data

    # Create an OperationTree instance
    def create(self, validated_data):
        quantity = None
        child_operation = None
        parent_operation1 = None
        parent_operation2 = None
        if "quantity" in validated_data:
            quantity = models.Quantity.objects.get(id=validated_data.pop("quantity"))
        if "child_operation" in validated_data:
            child_operation = models.OperationTree.objects.get(
                id=validated_data.pop("child_operation")
            )
        if not constant_or_variable(validated_data["operation"]):
            parent_operation1 = validated_data["parameters"].pop("a")
            parent_operation1["label"] = "a"
        if binary(validated_data["operation"]):
            parent_operation2 = validated_data["parameters"].pop("b")
            parent_operation2["label"] = "b"
        operation_tree = models.OperationTree.objects.create(
            quantity=quantity, child_operation=child_operation, **validated_data
        )
        if parent_operation1:
            parent_operation1["child_operation"] = operation_tree.id
            OperationTreeSerializer.create(
                OperationTreeSerializer(), validated_data=parent_operation1
            )
        if parent_operation2:
            parent_operation2["child_operation"] = operation_tree.id
            OperationTreeSerializer.create(
                OperationTreeSerializer(), validated_data=parent_operation2
            )
        return operation_tree


class QuantitySerializer(serializers.ModelSerializer):
    """Serialization, deserialization, and validation for the Quantity model."""

    operation_tree = OperationTreeSerializer(read_only=False, required=False)
    label = serializers.CharField(max_length=20)
    dataset = serializers.PrimaryKeyRelatedField(
        queryset=models.DataSet, required=False, allow_null=True
    )

    class Meta:
        model = models.Quantity
        fields = [
            "value",
            "variance",
            "units",
            "hash",
            "operation_tree",
            "label",
            "dataset",
            # "history",
        ]

    # TODO: should variable-only history be assumed to refer to the same Quantity and ignored?
    # Extract operation tree from history
    def to_internal_value(self, data):
        if "history" in data and "operation_tree" in data["history"]:
            operations = data["history"]["operation_tree"]
            if not operations["operation"] == "variable":
                data_copy = data.copy()
                data_copy["operation_tree"] = operations
                return super().to_internal_value(data_copy)
        return super().to_internal_value(data)

    # Serialize a Quantity instance
    def to_representation(self, instance):
        data = super().to_representation(instance)
        if "dataset" in data:
            data.pop("dataset")
        return data

    # Create a Quantity instance
    def create(self, validated_data):
        dataset = models.DataSet.objects.get(id=validated_data.pop("dataset"))
        if "operation_tree" in validated_data:
            operations_data = validated_data.pop("operation_tree")
            quantity = models.Quantity.objects.create(dataset=dataset, **validated_data)
            operations_data["quantity"] = quantity.id
            OperationTreeSerializer.create(
                OperationTreeSerializer(), validated_data=operations_data
            )
            return quantity
        else:
            return models.Quantity.objects.create(dataset=dataset, **validated_data)


class DataSetSerializer(serializers.ModelSerializer):
    """Serialization, deserialization, and validation for the DataSet model."""

    metadata = MetaDataSerializer(read_only=False)
    files = serializers.PrimaryKeyRelatedField(
        required=False, many=True, allow_null=True, queryset=models.DataFile
    )
    data_contents = QuantitySerializer(many=True, read_only=False)
    session = serializers.PrimaryKeyRelatedField(
        queryset=models.Session, required=False, allow_null=True
    )
    # TODO: handle files better

    class Meta:
        model = models.DataSet
        fields = [
            "id",
            "name",
            "files",
            "metadata",
            "data_contents",
            "is_public",
            "current_user",
            "users",
            "session",
        ]

    # Serialize a DataSet instance
    def to_representation(self, instance):
        data = super().to_representation(instance)
        if "session" in data:
            data.pop("session")
        return data

    # Check that private data has an owner
    def validate(self, data):
        if (
            not self.context["request"].user.is_authenticated
            and "is_public" in data
            and not data["is_public"]
        ):
            raise serializers.ValidationError("private data must have an owner")
        if "current_user" in data and data["current_user"] is None:
            if "is_public" in data:
                if not "is_public":
                    raise serializers.ValidationError("private data must have an owner")
            else:
                if not self.instance.is_public:
                    raise serializers.ValidationError("private data must have an owner")
        return data

    # Create a DataSet instance
    def create(self, validated_data):
        session = None
        if self.context["request"].user.is_authenticated:
            validated_data["current_user"] = self.context["request"].user
        metadata_raw = validated_data.pop("metadata")
        if session in validated_data:
            session = models.Session.objects.get(id=validated_data["session"])
        data_contents = validated_data.pop("data_contents")
        dataset = models.DataSet.objects.create(session=session, **validated_data)
        metadata_raw["dataset"] = dataset.id
        MetaDataSerializer.create(MetaDataSerializer(), validated_data=metadata_raw)
        for d in data_contents:
            d["dataset"] = dataset.id
            QuantitySerializer.create(QuantitySerializer(), validated_data=d)
        return dataset

    # TODO: account for updating other attributes
    # TODO: account for metadata potentially being null
    # Update a DataSet instance
    def update(self, instance, validated_data):
        if "metadata" in validated_data:
            metadata_raw = validated_data.pop("metadata")
            new_metadata = MetaDataSerializer.update(
                MetaDataSerializer(), instance.metadata, validated_data=metadata_raw
            )
            instance.metadata = new_metadata
            instance.save()
        instance.is_public = validated_data.get("is_public", instance.is_public)
        instance.name = validated_data.get("name", instance.name)
        instance.save()
        return instance

    # TODO: custom method for database to serializer representation


class PublishedStateSerializer(serializers.ModelSerializer):
    """Serialization, deserialization, and validation for the PublishedState model."""

    session = serializers.PrimaryKeyRelatedField(
        queryset=models.Session, required=False, allow_null=True
    )

    class Meta:
        model = models.PublishedState
        fields = "__all__"


class SessionSerializer(serializers.ModelSerializer):
    """Serialization, deserialization, and validation for the Session model."""

    datasets = DataSetSerializer(read_only=False, many=True)
    published_state = PublishedStateSerializer(read_only=False, required=False)

    class Meta:
        model = models.Session
        fields = ["title", "published_state", "datasets"]

    # Create a Session instance
    def create(self, validated_data):
        if self.context["request"].user.is_authenticated:
            validated_data["current_user"] = self.context["request"].user
        datasets = validated_data.pop("datasets")
        session = models.Session.objects.create(**validated_data)
        for dataset in datasets:
            dataset["session"] = session.id
            DataSetSerializer.create(DataSetSerializer(), validated_data=dataset)
        return session


# Determine if an operation does not have parent operations
def constant_or_variable(operation: str):
    return operation in ["zero", "one", "constant", "variable"]


# Determine if an operation has two parent operations
def binary(operation: str):
    return operation in ["add", "sub", "mul", "div", "dot", "matmul", "tensor_product"]
