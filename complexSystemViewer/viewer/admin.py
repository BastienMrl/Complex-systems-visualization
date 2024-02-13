from django.contrib import admin
from .models import ConfigurationItem, ColorParameter, NumberParameter, SelectionParameter, NumberRangeParameter, ALifeModel, Tool

# Register your models here.
admin.site.register(ConfigurationItem)
admin.site.register(ColorParameter)
admin.site.register(NumberParameter)
admin.site.register(SelectionParameter)
admin.site.register(NumberRangeParameter)
admin.site.register(ALifeModel)
admin.site.register(Tool)