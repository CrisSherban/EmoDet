from django.shortcuts import render
from django.http import HttpResponse
from .models import Person, PlotStats
from django.db.models.aggregates import Max
from django.db.models import Q


# Create your views here.

def get_latest_faces():
    latest_faces = Person.objects.values('person_number').annotate(person_last_seen=Max("person_last_seen"))

    q_statement = Q()
    for pair in latest_faces:
        q_statement |= (Q(person_number=pair['person_number']) & Q(person_last_seen=pair['person_last_seen']))

    return Person.objects.filter(q_statement).order_by('person_number')


def homepage(request):
    return render(request=request,
                  template_name="main/home.html",
                  context={'persons': Person.objects.all(),
                           'stats': PlotStats.objects.all(),
                           'latest_faces': get_latest_faces()
                           })


def info(request):
    return render(request=request,
                  template_name="main/info.html",
                  context={'persons': Person.objects.all().order_by('-person_last_seen')}
                  )

def about(request):
    return render(request=request,
                  template_name="main/about.html"
                  )
