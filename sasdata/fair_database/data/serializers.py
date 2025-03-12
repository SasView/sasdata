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
    metadata = MetaDataSerializer(read_only=False)
    files = serializers.PrimaryKeyRelatedField(
        required=False, many=True, allow_null=True, queryset=DataFile
    )

    class Meta:
        model = DataSet
        fields = "__all__"

    def create(self, validated_data):
        if self.context.user.is_authenticated:
            validated_data["current_user"] = self.context.user
        metadata_raw = validated_data.pop("metadata")
        metadata = MetaDataSerializer.create(
            MetaDataSerializer(), validated_data=metadata_raw
        )
        dataset = DataSet.objects.create(metadata=metadata, **validated_data)
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


class QuantitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Quantity
        fields = "__all__"


class OperationTreeSerializer(serializers.ModelSerializer):
    class Meta:
        model = OperationTree
        fields = "__all__"
