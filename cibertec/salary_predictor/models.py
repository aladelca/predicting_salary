from django.db import models


class Prediction(models.Model):
    description = models.TextField()
    skills_desc = models.TextField()
    title = models.CharField(max_length=255)
    predicted_salary = models.FloatField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.title} ({self.predicted_salary:.2f})"
