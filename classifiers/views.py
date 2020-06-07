from django.shortcuts import get_object_or_404, render

from .models import Classifier

def index(request):
    latest_classifier_list = Classifier.objects.order_by('-id')[:5]
    context = {
        'latest_classifier_list': latest_classifier_list,
    }
    return render(request, 'classifiers/index.html', context)

def detail(request, classifier_id):
    classifier_ = get_object_or_404(Classifier, pk=classifier_id)
    return render(request, 'classifiers/detail.html', {'classifier': classifier_})