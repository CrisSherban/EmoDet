from django import forms
from .models import Person

CHOICES = (
    ("ASC", "ascending"),
    ("DES", "descending")
)

EMOTIONS = (
    ("anger", "anger"),
    ("disgust", "disgust"),
    ("fear", "fear"),
    ("happy", "happy"),
    ("neutral", "neutral"),
    ("sadness", "sadness"),
    ("surprised", "surprised")
)


class PersonForm(forms.Form):
    prob_ordering = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=CHOICES,
        required=False,
        label="Order by Probability",
    )

    date_ordering = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=CHOICES,
        required=False,
        label="Order by Date")

    frame_number = forms.IntegerField(
        required=False,
        label="Frame of the face")

    emotion = forms.MultipleChoiceField(
        widget=forms.CheckboxSelectMultiple,
        choices=EMOTIONS,
        required=False,
        label="Emotion"
    )

    class Meta:
        model = Person
        fields = ('prob_ordering', 'date_ordering', 'frame_number', 'emotion')
