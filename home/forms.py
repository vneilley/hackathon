from django import forms
from .models import Stuff


class MainForm(forms.Form):
    class Meta:
        model = Stuff
        fields = ['weather_date', ' HighTemp','LowTemp', ' Rain', ' Snow' ]
    # weather_date = forms.DateField()
    # HighTemp = forms.FloatField()
    # LowTemp = forms.FloatField()
    # Rain = forms.FloatField()
    # Snow = forms.FloatField()