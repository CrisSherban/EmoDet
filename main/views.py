from django.shortcuts import render
from django.http import HttpResponse, HttpResponseRedirect
from .models import Person, PlotStats, AIPrediction
from django.db.models.aggregates import Max
from django.db.models import Q
from .forms import PersonForm
import os


def homepage(request):
    def get_latest_faces():
        latest_faces = Person.objects.values('person_number_in_last_frame') \
            .annotate(person_frame=Max("person_frame"))

        q_statement = Q()
        for pair in latest_faces:
            q_statement |= (Q(person_number_in_last_frame=pair['person_number_in_last_frame'])
                            & Q(person_frame=pair['person_frame']))

        return Person.objects.filter(q_statement)

    if request.method == 'POST' and 'record_cam' in request.POST:
        os.system("python AI.py cam")
        # this return avoids the RePOST after a page refresh
        return HttpResponseRedirect('/')

    if request.method == 'POST' and 'record_test' in request.POST:
        os.system("python AI.py no_cam")
        return HttpResponseRedirect('/')

    if request.method == 'POST' and 'purge' in request.POST:
        import purge_data
        purge_data.purge()
        return HttpResponseRedirect('/')

    return render(request=request,
                  template_name="main/home.html",
                  context={'persons': Person.objects.all(),
                           'stats': PlotStats.objects.all(),
                           'latest_faces': get_latest_faces()
                           })


def data(request):
    persons = Person.objects.all()

    if request.method == 'POST':
        form = PersonForm(request.POST)
        if form.is_valid():
            prob_ordering = form.cleaned_data['prob_ordering']
            date_ordering = form.cleaned_data['date_ordering']
            frame_number = form.cleaned_data['frame_number']
            emotion = form.cleaned_data['emotion']

            if len(prob_ordering) != 0 and len(date_ordering) != 0:
                if date_ordering[0] == 'ASC' and prob_ordering[0] == 'ASC':
                    persons = persons.order_by('person_last_seen', 'person_prediction_prob')
                if date_ordering[0] == 'ASC' and prob_ordering[0] == 'DES':
                    persons = persons.order_by('person_last_seen', '-person_prediction_prob')
                if date_ordering[0] == 'DES' and prob_ordering[0] == 'ASC':
                    persons = persons.order_by('-person_last_seen', 'person_prediction_prob')
                if date_ordering[0] == 'DES' and prob_ordering[0] == 'DES':
                    persons = persons.order_by('-person_last_seen', '-person_prediction_prob')

            if len(prob_ordering) == 0 and len(date_ordering) != 0:
                if date_ordering[0] == 'ASC':
                    persons = persons.order_by('person_last_seen')
                elif date_ordering[0] == 'DES':
                    persons = persons.order_by('-person_last_seen')

            if len(prob_ordering) != 0 and len(date_ordering) == 0:
                if prob_ordering[0] == 'ASC':
                    persons = persons.order_by('person_prediction_prob')
                elif prob_ordering[0] == 'DES':
                    persons = persons.order_by('-person_prediction_prob')

            if frame_number is not None:
                persons = persons.filter(person_frame=frame_number)

            if len(emotion) != 0:
                tmp_pers = Person.objects.none()
                for i in range(len(emotion)):
                    tmp_pers |= persons.filter(person_emotion=str(emotion[i]))
                persons = tmp_pers

    else:
        form = PersonForm()

    return render(request=request,
                  template_name="main/data.html",
                  context={
                      'persons': persons,
                      'form': form
                  })


def about(request):
    return render(request=request,
                  template_name="main/about.html"
                  )
