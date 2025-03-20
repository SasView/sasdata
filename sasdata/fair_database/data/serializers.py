from rest_framework import serializers

from data.models import DataFile, DataSet, MetaData, OperationTree, Quantity


class DataFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataFile
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
        model = MetaData
        fields = "__all__"


class DataSetSerializer(serializers.ModelSerializer):
    metadata = MetaDataSerializer(read_only=False)
    files = serializers.PrimaryKeyRelatedField(
        required=False, many=True, allow_null=True, queryset=DataFile
    )
    data_contents = serializers.DictField()
    # TODO: handle files better
    # TODO: see if I can find a better way to handle the quantity part

    class Meta:
        model = DataSet
        fields = ["name", "files", "metadata"]

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
        dataset = DataSet.objects.create(metadata=metadata, **validated_data)
        for d in data_contents:
            serializer = QuantitySerializer(data=data_contents[d])
            if serializer.is_valid():
                quantity = serializer.save()
                dataset.data_contents.add(quantity, through_defaults={"label": d})
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


class QuantitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Quantity
        fields = "__all__"


def constant_or_variable(operation: str):
    return str in ["zero", "one", "constant", "variable"]


def binary(operation: str):
    return str in ["add", "sub", "mul", "div", "dot", "matmul", "tensor_product"]


class OperationTreeSerializer(serializers.ModelSerializer):
    class Meta:
        model = OperationTree
        fields = ["dataset", "operation", "parameters"]

    def create(self, validated_data):
        parent_operation1 = None
        parent_operation2 = None
        if not constant_or_variable(validated_data["operation"]):
            parent1 = validated_data["parameters"].pop("a")
            parent1["dataset"] = validated_data["dataset"]
            serializer1 = OperationTreeSerializer(data=parent1)
            if serializer1.is_valid(raise_exception=True):
                parent_operation1 = serializer1.save()
        if binary(validated_data["operation"]):
            parent2 = validated_data["parameters"].pop("b")
            parent2["dataset"] = validated_data["dataset"]
            serializer2 = OperationTreeSerializer(data=parent2)
            if serializer2.is_valid(raise_exception=True):
                parent_operation2 = serializer2.save()
        return OperationTree.objects.create(
            dataset=validated_data["dataset"],  # TODO: check uuid vs object
            operation=validated_data["operation"],
            parameters=validated_data["parameters"],
            parent_operation1=parent_operation1,
            parent_operaton2=parent_operation2,
        )
