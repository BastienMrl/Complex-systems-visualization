from django.shortcuts import render
from .models import ALifeModel, TransformerItem, Parameter, Tool
from simulation.models.game_of_life import GOLSimulation
from simulation.modelManager import ModelEnum, ModelManager

# Create your views here.
def index(request):
    models = ALifeModel.objects.all()
    modelSelected = models.first()
    modelsName = [m.value for m in ModelEnum]
    print(modelsName)
    
    toolsList = Tool.objects.filter(aLifeModel=modelSelected.pk)
    
    rules = GOLSimulation.default_parameters
    rulesParameters = [rule.get_param() for rule in rules]
    
    transformers = TransformerItem.objects.filter(aLifeModel=modelSelected.pk)
    transformersParam = {}
    for t in transformers:
        param = Parameter.objects.filter(transformer=t.pk).select_subclasses()
        transformersParam[t] = param
    return render(request, "index.html", {"model":modelSelected , "modelsName":modelsName, "rulesParameters":rulesParameters, "transformers":transformersParam, "toolsList":toolsList}) 

def addTransformer(request, modelsName, transformerType ):
    model = ALifeModel.objects.filter(name=modelsName).first()
    baseTransformer = TransformerItem.objects.filter(aLifeModel=model, transformerType=transformerType).first()
    param = Parameter.objects.filter(transformer=baseTransformer.id).select_subclasses()
    return render(request, "transformers/transformerItem.html", {"transformer":baseTransformer, "parameters":param, "model":model})

def changeModel(request, modelsName):
    rules = ModelManager.get_default_params(modelsName)
    rulesParameters = [rule.get_param() for rule in rules]
    return render(request, "rulesItem.html", {"rules":rules, "rulesParameters":rulesParameters})