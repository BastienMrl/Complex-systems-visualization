from django.db import models

class ConfigurationItem(models.Model):
    name = models.CharField(max_length=128)
    
    def __str__(self):
        return self.name
    
class Parameter(models.Model):
    name = models.CharField(max_length=128)
    configuration = models.ForeignKey(ConfigurationItem, on_delete=models.CASCADE)
    class ParamType(models.TextChoices):
        INTRANGE = "IR"
    type = models.CharField(max_length=3,choices=ParamType,default=ParamType.INTRANGE)
    
    class Meta:
        abstract = True
        
class IntRangeParameter(Parameter):
    minDefaultValue = models.IntegerField()
    maxDefaultValue = models.IntegerField()