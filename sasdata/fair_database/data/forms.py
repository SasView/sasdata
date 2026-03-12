from data.models import DataFile
from django import forms


# Create the form class.
class DataFileForm(forms.ModelForm):
    class Meta:
        model = DataFile
        fields = ["file", "is_public"]
