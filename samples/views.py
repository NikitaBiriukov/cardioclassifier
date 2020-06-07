from django.shortcuts import get_object_or_404, render

from .models import Sample

def index(request):
    latest_sample_list = Sample.objects.order_by('id')[:5]
    context = {
        'latest_sample_list': latest_sample_list,
    }
    return render(request, 'samples/index.html', context)

def detail(request, sample_id):
    sample_ = get_object_or_404(Sample, pk=sample_id)
    return render(request, 'samples/detail.html', {'sample': sample_})