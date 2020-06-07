from django.contrib import admin

from .models import Sample
from .models import SampleType
from .models import SampleParameter

admin.site.register(Sample)
admin.site.register(SampleType)
admin.site.register(SampleParameter)
