from __future__ import unicode_literals

from django.db import models

# Create your models here.

class Stuff(models.Model):
    weather_date = models.DateField()
    HighTemp = models.FloatField()
    LowTemp = models.FloatField()
    Rain = models.FloatField()
    Snow = models.FloatField()
    TotalPatients = models.FloatField()


