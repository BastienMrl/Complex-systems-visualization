from django.urls import path

from . import views

urlpatterns = [
    path("", views.index, name="index"),
    path("addTranformerURL/<modelsName>/<transformerType>", views.addTransformer, name="addTransformer"),
    path("changeModel/<modelsName>", views.changeModel, name="changeModel")
]