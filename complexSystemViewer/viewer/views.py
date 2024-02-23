from django.shortcuts import render
from .models import ALifeModel, TransformerItem, RulesConfiguration, Parameter, Tool

# Create your views here.
def index(request):
    models = ALifeModel.objects.all()
    modelSelected = models.first()
    modelsName = [m.name for m in models]
    
    toolsList = Tool.objects.filter(aLifeModel=modelSelected.pk)
    
    rules = RulesConfiguration.objects.filter(aLifeModel=modelSelected.pk).first()
    rulesParameters = Parameter.objects.filter(configurationItem=rules.pk).select_subclasses()
    
    transformers = TransformerItem.objects.filter(aLifeModel=modelSelected.pk)
    transformersParam = {}
    for t in transformers:
        param = Parameter.objects.filter(configurationItem=t.id).select_subclasses()
        transformersParam[t] = param
    return render(request, "index.html", {"model":modelSelected , "modelsName":modelsName, "rulesParameters":rulesParameters, "transformers":transformersParam}) 

def addTransformer(request, modelsName, transformerType ):
    model = ALifeModel.objects.filter(name=modelsName).first()
    baseTransformer = TransformerItem.objects.filter(aLifeModel=model, transformerType=transformerType).first()
    param = Parameter.objects.filter(configurationItem=baseTransformer.id).select_subclasses()
    return render(request, "transformers/transformerItem.html", {"transformer":baseTransformer, "parameters":param, "model":model})
