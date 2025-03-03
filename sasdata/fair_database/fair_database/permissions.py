from rest_framework.permissions import BasePermission


def is_owner(request, obj):
    return request.user.is_authenticated and request.user == obj.current_user


def has_access(request, obj):
    return (
        is_owner(request, obj)
        or request.user.is_authenticated
        and request.user in obj.users.all()
    )


class DataPermission(BasePermission):
    def has_object_permission(self, request, view, obj):
        if request.method == "GET":
            if obj.is_public or has_access(request, obj):
                return True
        elif request.method == "DELETE":
            if obj.is_private and is_owner(request, obj):
                return True
        else:
            return is_owner(request, obj)


def check_permissions(request, obj):
    return DataPermission().has_object_permission(request, None, obj)
