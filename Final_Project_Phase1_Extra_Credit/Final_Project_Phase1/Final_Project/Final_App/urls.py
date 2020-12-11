from Final_App import views
from django.urls import path
from django.conf import settings
from django.conf.urls.static import static

app_name = 'Final_App'
urlpatterns = [
    path('', views.YourViewName.as_view(), name=app_name),
    #path('index/<int:pk>/', views.delete_file, name='delete_file'),
    
]
if settings.DEBUG:
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)
