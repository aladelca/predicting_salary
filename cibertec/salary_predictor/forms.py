from django import forms


class SalaryPredictionForm(forms.Form):
    description = forms.CharField(
        label="Job Description",
        widget=forms.Textarea(attrs={"rows": 6}),
        required=True,
    )
    skills_desc = forms.CharField(
        label="Skills Description",
        widget=forms.Textarea(attrs={"rows": 4}),
        required=True,
    )
    title = forms.CharField(
        label="Job Title",
        widget=forms.TextInput(),
        required=True,
    )
