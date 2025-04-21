from django.contrib import admin
from data import models

admin.site.register(models.DataFile)
admin.site.register(models.Session)
admin.site.register(models.PublishedState)
admin.site.register(models.DataSet)
admin.site.register(models.MetaData)
admin.site.register(models.Quantity)
admin.site.register(models.OperationTree)
admin.site.register(models.ReferenceQuantity)
