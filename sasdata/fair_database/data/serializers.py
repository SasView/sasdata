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
    # TODO: custom validation, maybe custom serialization handling of current_user
    # TODO: account for nested serialization
    metadata = MetaDataSerializer()

    class Meta:
        model = DataSet
        fields = "__all__"

    def create(self, validated_data):
        metadata_raw = validated_data.pop("metadata")
        metadata = MetaDataSerializer.create(
            MetaDataSerializer(), validated_data=metadata_raw
        )
        dataset = DataSet.objects.update_or_create(**validated_data, metadata=metadata)
        return dataset


class QuantitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Quantity
        fields = "__all__"


class OperationTreeSerializer(serializers.ModelSerializer):
    class Meta:
        model = OperationTree
        fields = "__all__"
