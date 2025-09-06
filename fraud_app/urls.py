from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path("upload_csv/", views.upload_csv, name="upload_csv"),
]
