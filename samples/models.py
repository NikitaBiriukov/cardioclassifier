import datetime

from django.db import models
from django.utils import timezone

class SampleType(models.Model):
    title = models.CharField(max_length=127)

#    class Meta:
#        app_label = 'sample_types'

    def __str__(self):
        return self.title


class Sample(models.Model):
    sample_type = models.ForeignKey(SampleType, on_delete=models.CASCADE)
    title = models.CharField(max_length=127)
    pub_date = models.DateTimeField('date published')

#    class Meta:
#        app_label = 'samples'

    def __str__(self):
        return self.title

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


class SampleParameter(models.Model):
    sample_type = models.ForeignKey(SampleType, on_delete=models.CASCADE)
    name = models.CharField(max_length=63)
    parameter_type = models.IntegerField(default=0) #string, int, float, bool

#    class Meta:
#        app_label = 'sample_parameters'

    def __str__(self):
        return self.name