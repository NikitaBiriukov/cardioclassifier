import datetime

from django.db import models
from django.utils import timezone
from classifiers.models import Classifier

class Task(models.Model):
    uuid = models.CharField(max_length=16) # unique identifier, used to store file
    pub_date = models.DateTimeField('date published')

#    class Meta:
#        app_label = 'tasks'

    def __str__(self):
        return self.uuid

    def was_published_recently(self):
        return self.pub_date >= timezone.now() - datetime.timedelta(days=1)


class TaskResult(models.Model):
    task_id = models.ForeignKey(Task, on_delete=models.CASCADE)
    uuid = models.CharField(max_length=16) # unique identifier, used to store file
    classifier_id = models.ForeignKey(Classifier, on_delete=models.CASCADE)

#    class Meta:
#        app_label = 'task_results'

    def __str__(self):
        return self.uuid