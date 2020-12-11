from django.db import models
import os
from django.utils import timezone


# Create your models here.

class FileModel(models.Model):
    file_name = models.CharField(max_length=50)
    file_content = models.FileField(upload_to="upload")

    def __str__(self):
        return self.file_name

    # def delete(self, *args, **kwargs):
    #     self.file_name.delete()
    #     self.file_content.delete()
    #     super().delete(*args, **kwargs)

    # def filename(self):
    #     return os.path.basename(self.file_name)

class AlgorithmModel(models.Model):
    algorithm_name = models.CharField(max_length=50)
    inference_script = models.FileField(upload_to='algorithms/scripts/')
    saved_model = models.FileField(upload_to='algorithms/saved_models/')

    def __str__(self):
        return self.algorithm_name

    def delete(self, *args, **kwargs):
        self.inference_script.delete()
        self.saved_model.delete()
        super().delete(*args, **kwargs)


class AnalyticModel(models.Model):
    analytic_name = models.CharField(max_length=50)
    result_plot = models.CharField(max_length=10000000)
    time = models.DateTimeField(default=timezone.now)
    DataSet_name = models.CharField(max_length=50)
    Algo_name = models.CharField(max_length=50)
