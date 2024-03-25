import os
from complexSystemViewer import settings
from django.shortcuts import render
from .models import ALifeModel, TransformerItem, Parameter, Tool
from simulation.models.game_of_life import GOLSimulation
from .simulationManager import SimulationEnum, SimulationManager

# Create your views here.
def index(request):
    models = ALifeModel.objects.all()
    modelSelected = models.first()
    modelsName = [m.value for m in SimulationEnum]
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

    meshPath = os.path.join(settings.BASE_DIR, "viewer/"+settings.STATIC_URL+"models/")
    meshFiles = os.listdir(meshPath)
    return render(request, "index.html", {"model":modelSelected , "modelsName":modelsName, "initParameters":initParameters,
                                          "rulesParameters":rulesParameters, "transformers":transformersParam, 
                                          "toolsList":toolsList, "meshFiles":meshFiles}) 

def addTransformer(request, transformerType):
    baseTransformer = TransformerItem.objects.filter(transformerType=transformerType).first()
    param = Parameter.objects.filter(transformer=baseTransformer).select_subclasses()
    return render(request, "visualizationPanel/transformers/transformerItem.html", {"transformer":baseTransformer, "parameters":param})

def changeModel(request, modelsName):
    rules = SimulationManager.get_default_rules(modelsName)
    rulesParameters = [rule.get_param() for rule in rules]
    
    init_p = SimulationManager.get_initialization_parameters(modelsName)
    initParameters = [ip.get_param() for ip in init_p]
    return render(request, "simulationPanel/simulationConfigSet.html", {"rulesParameters":rulesParameters, "initParameters":initParameters})