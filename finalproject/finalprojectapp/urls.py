from finalprojectapp import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static
app_name = 'finalprojectapp'
urlpatterns = [
path('', views.YourViewName.as_view(), name=app_name),
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)