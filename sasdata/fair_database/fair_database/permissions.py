from rest_framework.permissions import BasePermission


def is_owner(request, obj):
    return request.user.is_authenticated and request.user.id == obj.current_user


class DataPermission(BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method == "GET":
            if obj.is_public or is_owner(request, obj):
                return True
        elif request.method == "DELETE":
            if obj.is_private and is_owner(request, obj):
                return True
        else:
            return is_owner(request, obj)


def check_permissions(request, obj):
    if request.method == "GET":
        if obj.is_public or is_owner(request, obj):
            return True
    elif request.method == "DELETE":
        if obj.is_private and is_owner(request, obj):
            return True
    else:
        return is_owner(request, obj)
