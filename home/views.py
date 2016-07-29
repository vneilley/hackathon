from django.http import HttpResponse
from django.template import loader
from .models import Stuff
from .forms import MainForm
from django.shortcuts import render

def index(request):
    # return HttpResponse("<h1>homepage")

    if request.method ==  "Post":
        print (request.Post)


    form = MainForm()
    template = loader.get_template('home/index.html')
    context = {
        "form": form

    }
    # if request.method == "POST":
    #     # Get the posted form
    #     MyLoginForm = MainForm(request.POST)
    # else:
    #     result = MainForm()


    return render(request, "home/index.html", {})

def Result(request):
    template = loader.get_template('home/Result.html')
    return render(request, "home/Result.html", {})













