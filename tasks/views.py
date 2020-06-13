from django.shortcuts import get_object_or_404, render, redirect
from django.http import HttpResponse
from django.conf import settings

from .models import Task, TaskResult
from classifiers.models import Classifier

import uuid


import numpy as np
import pandas as pd
from sklearn import datasets, tree
import pydotplus


# knn
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix


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
    classifiers_ = Classifier.objects.all()
    return render(request, 'tasks/detail.html', {'task': task_, 'classifiers': classifiers_})

def results(request, task_id):
    response = "You're looking at the results of task %s."
    return HttpResponse(response % task_id)

def run(request, task_id):
    return HttpResponse("You're running task %s." % task_id)



# Decision Trees
def dtClassifier(X, y, feature_names, uuid_):
    # Create decision tree classifer object
    clf = tree.DecisionTreeClassifier(random_state=0)
    # Train model
    model = clf.fit(X, y)
    # Create DOT data
    dot_data = tree.export_graphviz(clf, out_file=None, feature_names=feature_names)#, class_names=target_names)
    # Draw graph
    graph = pydotplus.graph_from_dot_data(dot_data)
    # Create PDF
    graph.write_pdf(settings.MEDIA_ROOT + "/" + uuid_ + ".pdf")
    # Create PNG
    #graph.write_png(settings.MEDIA_ROOT + "/" + uuid_ + ".png")


# K-Nearest Neighbor
def knnClassifier(X, y, feature_names, uuid_):
    # Splitting the dataset into the Training set and Test set
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

    # Feature Scaling
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    # Fitting K-NN to the Training set
    classifier = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
    classifier.fit(X_train, y_train)

    # Predicting the Test set results
    y_pred = classifier.predict(X_test)

    # Making the Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)

    # Visualising the Training set results
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01), 
                            np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step =0.01))
    plt.contourf(X1, X2, 
                    classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), 
                    alpha = 0.75, 
                    cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

    plt.title('K-NN (Training set)')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    #   .show()

    # Visualising the Test set results
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:,0].max() + 1, step = 0.01),np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step =0.01))
    plt.contourf(X1,X2,classifier.predict(np.array([X1.ravel(),X2.ravel()]).T).reshape(X1.shape),alpha = 0.75, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(('red', 'green'))(i), label = j)

    plt.title('K-NN (Test set)')
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    plt.legend()
    #plt.show()
    plt.savefig(settings.MEDIA_ROOT + "/" + uuid_ + ".pdf")

# Support Vector Machine
def svmClassifier(X, y, feature_names, uuid_):
    dtClassifier(X, y, feature_names, uuid_)

# Radial Basis Function
def rbfClassifier(X, y, feature_names, uuid_):
    dtClassifier(X, y, feature_names, uuid_)

# Artificial Neural Network
def annClassifier(X, y, feature_names, uuid_):
    dtClassifier(X, y, feature_names, uuid_)

# Naive Bayes
def nbClassifier(X, y, feature_names, uuid_):
    dtClassifier(X, y, feature_names, uuid_)

def solver(request, task_id):
    classifier = Classifier.objects.get(id=request.POST['classifier_id'])
    task = Task.objects.get(id=task_id)
    tResult = TaskResult( task_id=task, uuid=str(uuid.uuid1()), classifier_id=classifier)
    tResult.save()
    # Importing the dataset
    fname = "cardioclassifier/media/cleveland.csv"
    feature_names = pd.read_csv(fname, index_col=0, nrows=0).columns.tolist()
    feature_names.pop()
    #feature_names = ["age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach", "exang", "oldpeak", "slope", "ca", "thal"]
    n_features = len(feature_names)
    target_names = ["num"]
    dataset = pd.read_csv(fname)
    # Separate the data from the target attributes
    X = dataset.iloc[:, 0:n_features].values
    y = dataset.iloc[:, n_features].values

    # Decision Trees
    if classifier.id == 1 :
        dtClassifier(X, y, feature_names, tResult.uuid)
    # K-Nearest Neighbor
    if classifier.id == 2 :
        knnClassifier(dataset.iloc[:, 0:2].values, dataset.iloc[:, n_features].values, feature_names, tResult.uuid)
    # Support Vector Machine
    if classifier.id == 3 :
        svmClassifier(X, y, feature_names, tResult.uuid)
    # Radial Basis Function
    if classifier.id == 4 :
        rbfClassifier(X, y, feature_names, tResult.uuid)
    # Artificial Neural Network
    if classifier.id == 5 :
        annClassifier(X, y, feature_names, tResult.uuid)
    # Naive Bayes
    if classifier.id == 6 :
        nbClassifier(X, y, feature_names, tResult.uuid)

    return redirect("/tasks/" + str(task_id))