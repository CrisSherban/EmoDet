from django.db import models


# Create your models here.
# Nice to know: each model maps to a database table


class Person(models.Model):
    person_id = models.IntegerField(primary_key=True, default=0)
    person_number_in_last_frame = models.IntegerField(default=0)
    person_frame = models.IntegerField(default=0)
    person_emotion = models.CharField(null=True, max_length=20)
    person_last_seen = models.DateTimeField('date_of_emotion', null=True)
    person_prediction_prob = models.FloatField(null=True)
    person_thumbnail = models.ImageField(upload_to="pics/",
                                         blank=True, null=True)

    def __str__(self):
        return self.person_id


class AIPrediction(models.Model):
    person = models.OneToOneField(
        Person,
        on_delete=models.CASCADE,
        to_field='person_id',
        primary_key=True,
    )

    anger = models.FloatField()
    disgust = models.FloatField()
    fear = models.FloatField()
    happy = models.FloatField()
    neutral = models.FloatField()
    sadness = models.FloatField()
    surprised = models.FloatField()
    plot = models.ImageField(upload_to="pics/",
                             blank=True, null=True)


class PlotStats(models.Model):
    plot_id = models.IntegerField(primary_key=True)
    plot = models.ImageField(upload_to="pics/",
                             blank=True, null=True)
