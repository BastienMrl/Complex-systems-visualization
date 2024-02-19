from django.shortcuts import render
from .models import ALifeModel, ConfigurationItem, Parameter, Tool

# Create your views here.
def index(request):
    models = ALifeModel.objects.all()
    modelSelected = models.first()
    modelsName = [m.name for m in models]
    configurations = ConfigurationItem.objects.filter(aLifeModel=models.first().pk)
    
    paramByConfig = {}
    for config in configurations:
        parameters = Parameter.objects.filter(configuration=config.pk).select_subclasses()
        paramByConfig[config] = parameters.reverse()
        
    toolsList = Tool.objects.filter(aLifeModel=models.first().pk)
        
    return render(request, "index.html", {"model":modelSelected , "modelsName":modelsName, "configurations":paramByConfig, "toolsList":toolsList})