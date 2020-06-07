from django.contrib import admin

from .models import Task
from .models import TaskResult

admin.site.register(Task)
admin.site.register(TaskResult)
