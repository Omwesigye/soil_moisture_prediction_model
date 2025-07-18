import os
import sys
import subprocess
from celery import shared_task
from django.conf import settings
from accounts.models import MLModel
from django.core.files import File
from django.utils import timezone
from datetime import datetime

@shared_task
def retrain_model_task(model_id, data_path=None, model_dir=None, model_name=None):
    # Hardcoded paths as per user request
    script_path = os.path.join(settings.BASE_DIR, 'ml_models', 'train_model.py')
    data_path = r'C:\Users\User\OneDrive\Desktop\RECESS PROJECT\ml_models\cleaned_soil_moisture_dataset.csv'
    model_path = r'C:\Users\User\OneDrive\Desktop\RECESS PROJECT\ml_models\soil_moisture_model.keras'
    feature_scaler_path = os.path.join(settings.BASE_DIR, 'ml_models', 'feature_scaler.pkl')
    moisture_scaler_path = os.path.join(settings.BASE_DIR, 'ml_models', 'moisture_scaler.pkl')
    action_encoder_path = os.path.join(settings.BASE_DIR, 'ml_models', 'action_encoder.pkl')
    print(f"[DEBUG] MEDIA_ROOT: {settings.MEDIA_ROOT}")
    print(f"[DEBUG] Looking for model at: {model_path}")

    # Run the training script with the specified arguments
    result = subprocess.run([
        sys.executable,
        script_path,
        data_path,
        model_path
    ], capture_output=True, text=True)

    if not os.path.exists(model_path):
        print(f"ERROR: Model file does not exist at {model_path}")
        raise FileNotFoundError(f"Model file does not exist at {model_path}")

    try:
        from accounts.models import MLModel
        from django.core.files import File
        from django.utils import timezone
        model_obj = MLModel(
            name=os.path.basename(model_path),
            is_active=True,
            uploaded_at=timezone.now()
        )
        with open(model_path, 'rb') as f:
            # Save the model file to the FileField
            model_obj.model_file.save(os.path.basename(model_path), File(f), save=True)
        model_obj.save()
        print(f"MLModel record created and saved: {model_obj}")
        print(f"[DEBUG] Feature scaler path: {feature_scaler_path}")
        print(f"[DEBUG] Moisture scaler path: {moisture_scaler_path}")
        print(f"[DEBUG] Action encoder path: {action_encoder_path}")
    except Exception as e:
        print(f"Exception during model save: {e}")
        raise

    return {
        'stdout': result.stdout,
        'stderr': result.stderr,
        'returncode': result.returncode
    }