
from django.conf.urls import patterns, url;
from home import views

urlpatterns = patterns('',
    url(r'^index$', views.index, name='index'),
    url(r'^Result$', views.Result, name='Result'),

);
