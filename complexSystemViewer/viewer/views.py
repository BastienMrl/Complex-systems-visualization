from django.shortcuts import render
from .models import ALifeModel, TransformerItem, Parameter, Tool
from simulation.models.game_of_life import GOLSimulation
from .modelManager import ModelEnum, ModelManager

# Create your views here.
def index(request):
    models = ALifeModel.objects.all()
    modelSelected = models.first()
    modelsName = [m.value for m in ModelEnum]
    print(modelsName)
    
    toolsList = Tool.objects.filter(aLifeModel=modelSelected.pk)
    
    rules = GOLSimulation.default_rules
    rulesParameters = [rule.get_param() for rule in rules]
    
    init_p = GOLSimulation.initialization_parameters
    initParameters = [ip.get_param() for ip in init_p]
    
    transformers = TransformerItem.objects.all()
    transformersParam = {}
    for t in transformers:
        param = Parameter.objects.filter(transformer=t.pk).select_subclasses()
        transformersParam[t] = param
    return render(request, "index.html", {"model":modelSelected , "modelsName":modelsName, "initParameters":initParameters ,"rulesParameters":rulesParameters, "transformers":transformersParam, "toolsList":toolsList}) 

def addTransformer(request, transformerType):
    baseTransformer = TransformerItem.objects.filter(transformerType=transformerType).first()
    param = Parameter.objects.filter(transformer=baseTransformer).select_subclasses()
    return render(request, "transformers/transformerItem.html", {"transformer":baseTransformer, "parameters":param})

def changeModel(request, modelsName):
    rules = ModelManager.get_default_rules(modelsName)
    rulesParameters = [rule.get_param() for rule in rules]
    
    init_p = ModelManager.get_initialization_parameters(modelsName)
    initParameters = [ip.get_param() for ip in init_p]
    return render(request, "simulationConfigSet.html", {"rulesParameters":rulesParameters, "initParameters":initParameters})