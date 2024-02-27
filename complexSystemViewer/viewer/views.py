from django.shortcuts import render
from .models import ALifeModel, TransformerItem, RulesConfiguration, Parameter, Tool
from simulation.models.game_of_life import GOLSimulation

# Create your views here.
def index(request):
    models = ALifeModel.objects.all()
    modelSelected = models.first()
    modelsName = [m.name for m in models]
    
    #toolsList = Tool.objects.filter(aLifeModel=models.first().pk)
    
    # rules = RulesConfiguration.objects.filter(aLifeModel=modelSelected.pk).first()
    # rulesParameters = Parameter.objects.filter(configurationItem=rules.pk).select_subclasses()
    rules = GOLSimulation.default_parameters
    rulesParameters = [rule.get_param() for rule in rules]
    
    
    transformers = TransformerItem.objects.filter(aLifeModel=modelSelected.pk)
    transformersParam = {}
    for t in transformers:
        param = Parameter.objects.filter(configurationItem=t.id).select_subclasses()
        transformersParam[t] = param
        
    return render(request, "index.html", {"model":modelSelected , "modelsName":modelsName, "rulesParameters":rulesParameters, "transformers":transformersParam})