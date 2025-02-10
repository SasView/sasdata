from rest_framework import serializers

from data.models import DataFile

class DataFileSerializer(serializers.ModelSerializer):
    class Meta:
        model = DataFile
        fields = "__all__"

    def validate(self, data):
        if not self.context['is_public'] and not data['current_user']:
            raise serializers.ValidationError('private data must have an owner')
        return data