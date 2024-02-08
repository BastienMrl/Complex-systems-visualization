from django.shortcuts import render
from .models import ALifeModel, ConfigurationItem, Parameter

# Create your views here.
def index(request):
    models = ALifeModel.objects.all()
    configurations = ConfigurationItem.objects.filter(aLifeModel=models.first().pk)
    
    paramByConfig = {}
    for config in configurations:
        parameters = Parameter.objects.filter(configuration=config.pk).select_subclasses()
        paramByConfig[config] = parameters
        
    return render(request, "index.html", {"models":models, "configurations":paramByConfig})