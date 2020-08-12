from django.db import models


# Create your models here.
# Nice to know: each model maps to a database table


class Person(models.Model):
    person_number = models.IntegerField(primary_key=True)
    person_emotion = models.CharField(max_length=20)
    person_last_seen = models.DateTimeField('date_of_emotion', null=True)
    person_prediction_prob = models.FloatField(null=True)
    person_thumbnail = models.ImageField(upload_to="faces_pics/",
                                         blank=True, null=True)

    def __str__(self):
        return str(self.person_number)


class PlotStats(models.Model):
    plot_id = models.IntegerField(primary_key=True)
    plot = models.ImageField(upload_to="faces_pics/",
                             blank=True, null=True)
