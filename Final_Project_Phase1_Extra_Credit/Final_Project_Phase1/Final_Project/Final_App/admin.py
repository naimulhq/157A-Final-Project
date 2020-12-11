from django.contrib import admin
from .models import AlgorithmModel
# admin.site.register(AlgorithmModel)

class AlgorithmAdmin(admin.ModelAdmin):
    pass

# Register the admin class with the associated model
admin.site.register(AlgorithmModel, AlgorithmAdmin)
