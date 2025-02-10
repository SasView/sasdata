import os

from rest_framework import serializers

from .models import Data

class DataSerializer(serializers.ModelSerializer):
    class Meta:
        model = Data
        fields = "__all__"

    def validate(self, data):
        if not self.context['is_public'] and not data['current_user']:
            raise serializers.ValidationError('private data must have an owner')
        return data