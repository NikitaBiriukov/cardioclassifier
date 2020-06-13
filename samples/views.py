from django.shortcuts import get_object_or_404, render

from .models import Sample, SampleType

def index(request):
    latest_sample_list = Sample.objects.order_by('id')[:5]
    sample_types = SampleType.objects.all() 
    context = {
        'latest_sample_list': latest_sample_list,
        'sample_types': sample_types
    }
    return render(request, 'samples/index.html', context)

def detail(request, sample_id):
    sample_ = get_object_or_404(Sample, pk=sample_id)
    return render(request, 'samples/detail.html', {'sample': sample_})