from django.contrib import admin
from .models import ALifeModel, TransformerItem, Tool, ColorParameter, NumberParameter, NumberRangeParameter, SelectionParameter

# Register your models here.
admin.site.register(ALifeModel)
admin.site.register(TransformerItem)
admin.site.register(Tool)
admin.site.register(ColorParameter)
admin.site.register(NumberParameter)
admin.site.register(NumberRangeParameter)
admin.site.register(SelectionParameter)