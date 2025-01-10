from django.db import models
from django.utils import timezone

# Create your models here.




class FaceEmbedding(models.Model):
    person_name = models.CharField(max_length=255)
    role = models.CharField(max_length=100)
    embedding = models.BinaryField()
    registered_at = models.DateTimeField(default=timezone.now)  # Manually setting the default
    face_image = models.ImageField(upload_to='faces/', null=True, blank=True)  # Optional

    def __str__(self):
        return f"{self.person_name} - {self.role}"
