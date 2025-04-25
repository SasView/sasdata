from rest_framework.permissions import BasePermission


# check if a request is made by an object's owner
def is_owner(request, obj):
    return request.user.is_authenticated and request.user == obj.current_user


# check if a request is made by a user with read access
def has_access(request, obj):
    return is_owner(request, obj) or (
        request.user.is_authenticated and request.user in obj.users.all()
    )


class DataPermission(BasePermission):
    # check if a request has the correct permissions for a specific object
    def has_object_permission(self, request, view, obj):
        if request.method == "GET":
            return obj.is_public or has_access(request, obj)
        elif request.method == "DELETE":
            return not obj.is_public and is_owner(request, obj)
        else:
            return is_owner(request, obj)


# check if a request has the correct permissions for a specific object
def check_permissions(request, obj):
    return DataPermission().has_object_permission(request, None, obj)
