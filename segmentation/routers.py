from rest_framework_nested.routers import NestedMixin, SimpleRouter
from rest_framework.routers import Route, SimpleRouter


class BulkNestedRouter(NestedMixin, SimpleRouter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.routes += [
            Route(
                url=r'^{prefix}{trailing_slash}$',
                mapping={'get': 'list', 'post': 'create', 'delete': 'bulk_destroy'},
                name='{basename}-images',
                detail=False,
                initkwargs={'suffix': ''}
            )
        ]
