from django.contrib import admin
from .models import ALifeModel, Tool, RulesConfiguration, TransformerItem, ColorParameter, NumberParameter, NumberRangeParameter, SelectionParameter

# Register your models here.
admin.site.register(ALifeModel)
admin.site.register(Tool)
admin.site.register(RulesConfiguration)
admin.site.register(TransformerItem)
admin.site.register(ColorParameter)
admin.site.register(NumberParameter)
admin.site.register(NumberRangeParameter)
admin.site.register(SelectionParameter)