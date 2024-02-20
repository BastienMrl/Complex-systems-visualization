import uuid
from django.db import models
from model_utils.managers import InheritanceManager

class ALifeModel(models.Model):
    name = models.CharField(max_length=128)
    simOutType = models.CharField(max_length=512, help_text="Wrote types separate with '/'")
    def getSimOutType(self):
        return self.simOutType.split('/')
    def __str__(self):
        return self.name
    
class ConfigurationItem(models.Model):
    id = models.UUIDField(primary_key=True, default=uuid.uuid4, editable=False)
    idHtml = models.CharField(max_length=128)
    name = models.CharField(max_length=128)
    aLifeModel = models.ForeignKey(ALifeModel, on_delete=models.CASCADE)
    def __str__(self):
        return self.name + " of " + self.aLifeModel.name

class RulesConfiguration(ConfigurationItem):
    def __str__(self) :
        return "Rules of " + self.aLifeModel.__str__()
    def getIdHtml(self):
        return self.aLifeModel.__str__() + "srules"

class TransformerItem(ConfigurationItem):
    outputType = models.CharField(max_length=128, default="POSITION_X")
    transformerType = models.CharField(max_length=128, default="COLOR")
    def __str__(self):
        return self.name + " of " + self.aLifeModel.__str__()

class ParamType(models.TextChoices):
        NUMBERRANGE = "NR",
        NUMBERVALUE = "NV",
        COLORVALUE = "CV",
        SELECTIONVALUE = "SV"
    
class Parameter(models.Model):
    paramId = models.CharField(max_length=128, null=True)
    name = models.CharField(max_length=128)
    configurationItem = models.ForeignKey(ConfigurationItem, on_delete=models.CASCADE)
    objects = InheritanceManager()
    
    def __str__(self):
        return self.name + " from " + self.configurationItem.__str__()
        
class NumberRangeParameter(Parameter):
    minDefaultValue = models.FloatField()
    maxDefaultValue = models.FloatField()
    step = models.FloatField()
    type = ParamType.NUMBERRANGE
    
class NumberParameter(Parameter):
    defaultValue = models.FloatField()
    step = models.FloatField()
    minValue = models.FloatField(null=True, blank=True)
    maxValue = models.FloatField(null=True, blank=True)
    type = ParamType.NUMBERVALUE
    
class ColorParameter(Parameter):
    hexDefaultValue = models.CharField(max_length=7)
    type = ParamType.COLORVALUE
    
class SelectionParameter(Parameter):
    options = models.CharField(max_length=512, help_text="Wrote options separate with '/'")
    defaultValue = models.SmallIntegerField(default=0)
    type = ParamType.SELECTIONVALUE
    
    def getoptionsList(self):
        return self.options.split(sep='/')
    
class Tool(models.Model):
    name = models.CharField(max_length=128)
    description = models.CharField(max_length=256, null=True, blank=True)
    aLifeModel = models.ForeignKey(ALifeModel, on_delete=models.CASCADE)
    def __str__(self):
         return self.name + " of " + self.aLifeModel.name