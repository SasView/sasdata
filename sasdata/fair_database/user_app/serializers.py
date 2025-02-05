from rest_framework import serializers

from rest_auth.serializers import UserDetailsSerializer


class KnoxSerializer(serializers.Serializer):
    """
    Serializer for Knox authentication.
    """
    token = serializers.CharField()
    user = UserDetailsSerializer()