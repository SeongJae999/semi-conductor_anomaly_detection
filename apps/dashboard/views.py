from django.shortcuts import render
from django.http import HttpResponse

from django.shortcuts import render
import io

from django.shortcuts import render
from django.views.generic.edit import CreateView

# Create your views here.
def landing(request):
    return render(
        request,
        'dashboard/landing.html'
    )
    