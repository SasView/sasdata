from django import forms
from data.models import DataFile


# Create the form class.
class DataFileForm(forms.ModelForm):
    class Meta:
        model = DataFile
        fields = ["file", "is_public"]
