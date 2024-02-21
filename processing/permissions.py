from rest_framework import permissions
from django.http import HttpRequest
from django.db.models import Model
from django.views import View


class IsOwnerOrReadOnly(permissions.BasePermission):
    def has_object_permission(self, request: HttpRequest, view: View, obj: Model) -> bool:
        if obj.owner == request.user:
            return True
        elif obj.public and request.method in permissions.SAFE_METHODS:
            return True
        else:
            return False
