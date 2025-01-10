from django.urls import path
from .views import register

urlpatterns = [
    path('', register, name='register'),  # Redirect /embeddings to the registration view
    path('register/', register, name='register'),
]
