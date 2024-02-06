from rest_framework_nested.routers import NestedSimpleRouter
from rest_framework.routers import Route, SimpleRouter


class ImageBulkRouter(NestedSimpleRouter):
    routes = [
        Route(
            url=r'^{prefix}{trailing_slash}$',
            mapping={'get': 'list', 'post': 'create', 'delete': 'bulk_destroy_images'},
            name='{basename}-images',
            detail=False,
            initkwargs={'suffix': 'BulkCreate'}
        ),
        Route(
            url=r'^{prefix}/predictions{trailing_slash}$',
            mapping={'delete': 'bulk_destroy_predictions'},
            name='{basename}-predictions',
            detail=False,
            initkwargs={'suffix': 'BulkDelete'}
        ),
    ] + NestedSimpleRouter.routes + SimpleRouter.routes
