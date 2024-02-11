from django.urls import re_path
from . import consumers

websocket_urlpatterns = [
    re_path(r"ws/viewer/", consumers.ViewerConsumerV2.as_asgi()),
]