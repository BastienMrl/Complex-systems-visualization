from django.shortcuts import render
from .models import ConfigurationItem, IntRangeParameter

# Create your views here.
def index(request):
    parameters = IntRangeParameter.objects.all()
    return render(request, "index.html", {'parameters':parameters})