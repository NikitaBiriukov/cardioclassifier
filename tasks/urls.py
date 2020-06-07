from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

from . import views

app_name = 'tasks'
urlpatterns = [
    path('', views.index, name='index'),
    path('<int:task_id>/', views.detail, name='detail'),
    path('<int:task_id>/results/', views.results, name='results'),
    path('<int:task_id>/run/', views.run, name='run'),
    path('<int:task_id>/solver', views.solver, name='solver'),
] + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
