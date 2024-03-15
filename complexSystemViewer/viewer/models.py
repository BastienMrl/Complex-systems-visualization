import uuid
from django.db import models
from model_utils.managers import InheritanceManager

class ALifeModel(models.Model):
    name = models.CharField(max_length=128)
    simOutType = models.CharField(max_length=512, help_text="Wrote types separate with '/'")
    transformerType = models.CharField(max_length=512, help_text="Wrote types separate with '/'")
    def getSimOutType(self):
        return self.simOutType.split('/')
    def getTransformerType(self):
        return self.transformerType.split('/')
    def __str__(self):
        return self.name

class TransformerItem(models.Model):
    idHtml = models.CharField(max_length=128)
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=1024)
    outputType = models.CharField(max_length=128, default="POSITION_X")
    transformerType = models.CharField(max_length=128, default="COLOR")
    displayedByDefault = models.BooleanField(default=True)
    isDeletable = models.BooleanField(default=True)
    def __str__(self):
        return self.name
    
class Parameter(models.Model):
    paramId = models.CharField(max_length=128, null=True)
    name = models.CharField(max_length=128)
    transformer = models.ForeignKey(TransformerItem, on_delete=models.CASCADE)
    objects = InheritanceManager()
    
    def __str__(self):
        return self.name + " from " + self.transformer.__str__()
        
class NumberRangeParameter(Parameter):
    minDefaultValue = models.FloatField()
    maxDefaultValue = models.FloatField()
    step = models.FloatField()
    
    def type(self):
        return "NR"
    
class NumberParameter(Parameter):
    defaultValue = models.FloatField()
    step = models.FloatField()
    minValue = models.FloatField(null=True, blank=True)
    maxValue = models.FloatField(null=True, blank=True)
    
    def type(self):
        return "NV"
    
class ColorParameter(Parameter):
    hexDefaultValue = models.CharField(max_length=7)
    
    def type(self):
        return "CV"
    
class SelectionParameter(Parameter):
    options = models.CharField(max_length=512, help_text="Wrote options separate with '/'")
    defaultValue = models.SmallIntegerField(default=0)
    
    def getoptionsList(self):
        return self.options.split(sep='/')
    
    def type(self):
        return "SV"
    
class SelectionMode(models.TextChoices):
    BOX = "BOX"
    BRUSH = "BRUSH"
    
class Tool(models.Model):
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=256, null=True, blank=True)
    aLifeModel = models.ForeignKey(ALifeModel, on_delete=models.CASCADE)
    selectionMode = models.CharField(max_length=10,choices=SelectionMode)
    def __str__(self):
         return self.name + " of " + self.aLifeModel.name