from django.db import models

class Classifier(models.Model):
#    sample_type = models.ForeignKey(SampleType, on_delete=models.CASCADE)
    title = models.CharField(max_length=127)
    executable = models.CharField(max_length=127) # path to executable
    settings = models.CharField(max_length=1023) # command line settings
    description = models.CharField(max_length=2048) # short description

#    class Meta:
#        app_label = 'classifiers'

    def __str__(self):
        return self.title
