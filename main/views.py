from django.shortcuts import render
from django.http import HttpResponse
from .models import Person, PlotStats


# Create your views here.
def homepage(request):
    return render(request=request,
                  template_name="main/home.html",
                  context={'persons': Person.objects.all(),
                           'stats': PlotStats.objects.all()
                           })
