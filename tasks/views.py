from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponse
from django.conf import settings

from .models import Task, TaskResult
from classifiers.models import Classifier

from sklearn.tree import DecisionTreeClassifier
from sklearn import datasets, tree
import pydotplus

import uuid


def index(request):
    latest_task_list = Task.objects.order_by('-pub_date')[:5]
#    output = ', '.join([q.uuid for q in latest_task_list])
#    return HttpResponse(output)

#    template = loader.get_template('tasks/index.html')
#    context = {
#        'latest_task_list': latest_task_list,
#    }
#    return HttpResponse(template.render(context, request))
    context = {
        'latest_task_list': latest_task_list,
    }
    return render(request, 'tasks/index.html', context)



def detail(request, task_id):
    task_ = get_object_or_404(Task, pk=task_id)
    return render(request, 'tasks/detail.html', {'task': task_})

def results(request, task_id):
    response = "You're looking at the results of task %s."
    return HttpResponse(response % task_id)

def run(request, task_id):
    return HttpResponse("You're running task %s." % task_id)

def solver(request, task_id):
    classifier = Classifier.objects.get(id=1)
    task = Task.objects.get(id=task_id)
    tResult = TaskResult( task_id=task, uuid=str(uuid.uuid1()), classifier_id=classifier)
#    tResult.task_id = task_id
#    tResult.uuid = 
#    tResult.classifier_id_id = 1
    tResult.save()

    # Load data
    iris = datasets.load_iris()
    X = iris.data
    y = iris.target

    #print(iris.data)
    #print(iris.target)

    # Create decision tree classifer object
    clf = DecisionTreeClassifier(random_state=0)
    # Train model
    model = clf.fit(X, y)
    # Create DOT data
    dot_data = tree.export_graphviz(clf, out_file=None, 
                                    feature_names=iris.feature_names,  
                                    class_names=iris.target_names)
    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)  
    # Create PDF
    graph.write_pdf(settings.MEDIA_ROOT + "/" + tResult.uuid + ".pdf")
    # Create PNG
    graph.write_png(settings.MEDIA_ROOT + "/" + tResult.uuid + ".png")
    return redirect("/tasks/" + str(task_id))