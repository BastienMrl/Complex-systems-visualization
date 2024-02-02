from django.db import models

class ConfiguationItem(models.Model):
    name = models.CharField(max_length=128)
    
    def __str__(self):
        return self.name
    
class Parameter(models.Model):
    name = models.CharField(max_length=128)
    configuration = models.ForeignKey(ConfiguationItem, on_delete=models.CASCADE)
    
    class Meta:
        abstract = True
        
class IntRangeParameter(Parameter):
    minDefaultValue = models.IntegerField()
    maxDefaultValue = models.IntegerField()