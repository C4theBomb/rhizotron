from django.shortcuts import render


def starter_kit(request):
    context = {"breadcrumb": {"parent": "Color Version", "child": "Layout Light"}}
    return render(request, 'index.html', context)
