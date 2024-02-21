from django.views import View
from django.http import HttpResponse
from django.shortcuts import render


class Home(View):
    def get(self, request) -> HttpResponse:
        return render(request, 'core/index.html')
