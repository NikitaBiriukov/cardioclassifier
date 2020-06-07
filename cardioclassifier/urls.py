from django.contrib import admin
from django.urls import path, include
from django.conf import settings
from django.conf.urls.static import static

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('authentication.urls')),
    path('', include('app.urls')),
    path('classifiers/', include('classifiers.urls')),
    path('samples/', include('samples.urls')),
    path('tasks/', include('tasks.urls'))
]  + static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
