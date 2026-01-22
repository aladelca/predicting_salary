from pathlib import Path

from django.conf import settings
from django.core.management.base import BaseCommand

from salary_predictor.ml.train import train_model


class Command(BaseCommand):
    help = "Train the salary prediction model and save artifacts."

    def add_arguments(self, parser):
        parser.add_argument(
            "--data-path",
            default=str(settings.SALARY_DATA_PATH),
            help="Path to postings.csv",
        )
        parser.add_argument(
            "--artifacts-dir",
            default=str(settings.SALARY_MODEL_DIR),
            help="Directory to store model artifacts",
        )

    def handle(self, *args, **options):
        data_path = Path(options["data_path"])
        artifacts_dir = Path(options["artifacts_dir"])
        result = train_model(data_path, artifacts_dir)
        metrics = result["metrics"]
        self.stdout.write(self.style.SUCCESS("Training completed."))
        self.stdout.write(f"MAE: {metrics['mae']:.4f}")
        self.stdout.write(f"MAPE: {metrics['mape']:.4f}")
        self.stdout.write(
            f"Artifacts saved to: {result['artifacts']['model'].parent}"
        )
