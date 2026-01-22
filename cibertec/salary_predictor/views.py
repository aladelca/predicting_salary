from django.shortcuts import render

from salary_predictor.forms import SalaryPredictionForm
from salary_predictor.ml.predictor import predict_salary
from salary_predictor.models import Prediction


def home_view(request):
    predictions = Prediction.objects.order_by("-created_at")
    return render(
        request,
        "salary_predictor/home.html",
        {"predictions": predictions},
    )


def predict_view(request):
    prediction = None
    error = None

    if request.method == "POST":
        form = SalaryPredictionForm(request.POST)
        if form.is_valid():
            try:
                prediction = predict_salary(
                    form.cleaned_data["description"],
                    form.cleaned_data["skills_desc"],
                    form.cleaned_data["title"],
                )
                Prediction.objects.create(
                    description=form.cleaned_data["description"],
                    skills_desc=form.cleaned_data["skills_desc"],
                    title=form.cleaned_data["title"],
                    predicted_salary=prediction,
                )
            except FileNotFoundError as exc:
                error = str(exc)
            except Exception:
                error = "Prediction failed. Please try again."
    else:
        form = SalaryPredictionForm()

    return render(
        request,
        "salary_predictor/predict.html",
        {
            "form": form,
            "prediction": prediction,
            "error": error,
        },
    )
