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


class DataSetSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataSet
        fields = "__all__"


class QuantitySerializer(serializers.ModelSerializer):
    class Meta:
        model = Quantity
        fields = "__all__"


class MetaDataSerializer(serializers.ModelSerializer):
    class Meta:
        model = MetaData
        fields = "__all__"


class OperationTreeSerializer(serializers.ModelSerializer):
    class Meta:
        model = OperationTree
        fields = "__all__"
