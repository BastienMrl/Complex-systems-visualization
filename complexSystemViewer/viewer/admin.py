from django.contrib import admin
from .models import ConfigurationItem, IntRangeParameter

# Register your models here.
admin.site.register(ConfigurationItem)
admin.site.register(IntRangeParameter)