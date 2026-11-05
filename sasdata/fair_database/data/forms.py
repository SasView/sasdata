from django import forms

from sasdata.fair_database.data.models import DataFile


# Create the form class.
class DataFileForm(forms.ModelForm):
    class Meta:
        model = DataFile
        fields = ["file", "is_public"]
