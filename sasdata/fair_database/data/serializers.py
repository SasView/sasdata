from rest_framework import serializers

from data import models


class DataFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.DataFile
        fields = "__all__"

    def validate(self, data):
        if not self.context["is_public"] and not data["current_user"]:
            raise serializers.ValidationError("private data must have an owner")
        return data


class AccessManagementSerializer(serializers.Serializer):
    username = serializers.CharField(max_length=200, required=False)
    access = serializers.BooleanField()


class MetaDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.MetaData
        fields = "__all__"


class OperationTreeSerializer(serializers.ModelSerializer):
    class Meta:
        model = models.OperationTree
        fields = ["quantity", "operation", "parameters"]

    def create(self, validated_data):
        parent_operation1 = None
        parent_operation2 = None
        if not constant_or_variable(validated_data["operation"]):
            parent1 = validated_data["parameters"].pop("a")
            parent1["quantity"] = validated_data["quantity"]
            serializer1 = OperationTreeSerializer(data=parent1)
            if serializer1.is_valid(raise_exception=True):
                parent_operation1 = serializer1.save()
        if binary(validated_data["operation"]):
            parent2 = validated_data["parameters"].pop("b")
            parent2["quantity"] = validated_data["quantity"]
            serializer2 = OperationTreeSerializer(data=parent2)
            if serializer2.is_valid(raise_exception=True):
                parent_operation2 = serializer2.save()
        return models.OperationTree.objects.create(
            dataset=validated_data["quantity"],  # TODO: check uuid vs object
            operation=validated_data["operation"],
            parameters=validated_data["parameters"],
            parent_operation1=parent_operation1,
            parent_operaton2=parent_operation2,
        )


class QuantitySerializer(serializers.ModelSerializer):
    label = serializers.CharField(max_length=20)
    history = serializers.JSONField(required=False)  # TODO: is this required?

    class Meta:
        model = models.Quantity
        fields = ["value", "variance", "units", "hash", "label", "history"]

    # TODO: validation checks for history

    def create(self, validated_data):
        if "history" in validated_data:
            operations_raw = validated_data.pop("history")
            quantity = models.Quantity.objects.create(**validated_data)
            operations_tree_raw = operations_raw["operation_tree"]
            operations_tree_raw["quantity"] = quantity.id
            serializer = OperationTreeSerializer(data=operations_tree_raw)
            if serializer.is_valid():
                serializer.save()
            return quantity
        else:
            return models.Quantity.objects.create(**validated_data)


class DataSetSerializer(serializers.ModelSerializer):
    metadata = MetaDataSerializer(read_only=False)
    files = serializers.PrimaryKeyRelatedField(
        required=False, many=True, allow_null=True, queryset=models.DataFile
    )
    data_contents = QuantitySerializer(many=True, read_only=False)
    # TODO: handle files better
    # TODO: see if I can find a better way to handle the quantity part

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
        ]

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

    def create(self, validated_data):
        if self.context["request"].user.is_authenticated:
            validated_data["current_user"] = self.context["request"].user
        metadata_raw = validated_data.pop("metadata")
        metadata = MetaDataSerializer.create(
            MetaDataSerializer(), validated_data=metadata_raw
        )
        data_contents = validated_data.pop("data_contents")
        dataset = models.DataSet.objects.create(metadata=metadata, **validated_data)
        for d in data_contents:
            label = d.pop("label")
            quantity = QuantitySerializer.create(QuantitySerializer(), validated_data=d)
            dataset.data_contents.add(quantity, through_defaults={"label": label})
        return dataset

    # TODO: account for updating other attributes
    # TODO: account for metadata potentially being null
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
    class Meta:
        model = models.PublishedState
        fields = "__all__"


class SessionSerializer(serializers.ModelSerializer):
    dataset = DataSetSerializer(read_only=False, many=True)
    published_state = PublishedStateSerializer(read_only=False)

    class Meta:
        model = models.Session
        fields = "__all__"


def constant_or_variable(operation: str):
    return str in ["zero", "one", "constant", "variable"]


def binary(operation: str):
    return str in ["add", "sub", "mul", "div", "dot", "matmul", "tensor_product"]
