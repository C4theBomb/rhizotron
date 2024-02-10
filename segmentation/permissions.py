from rest_framework import permissions


class IsOwnerOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request, view, obj):
        if obj.owner == request.user:
            return True
        elif obj.public and request.method in permissions.SAFE_METHODS:
            return True
        else:
            return False
