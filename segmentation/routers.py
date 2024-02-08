from rest_framework_nested.routers import NestedSimpleRouter
from rest_framework.routers import Route, SimpleRouter


class BulkNestedRouter(NestedSimpleRouter):
    routes = [
        Route(
            url=r'^{prefix}{trailing_slash}$',
            mapping={'get': 'list', 'post': 'create', 'delete': 'bulk_destroy'},
            name='{basename}-images',
            detail=False,
            initkwargs={'suffix': ''}
        )
    ] + NestedSimpleRouter.routes + SimpleRouter.routes
