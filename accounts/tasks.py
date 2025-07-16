import os
import sys
import subprocess
from celery import shared_task
from django.conf import settings

@shared_task(bind=True)
def retrain_model_task(self, model_id, data_path, output_path):
    script_path = os.path.join(settings.BASE_DIR, 'ml_models', 'train_model.py')

    feature_scaler_path = os.path.join(settings.BASE_DIR, 'ml_models', 'feature_scaler.pkl')
    moisture_scaler_path = os.path.join(settings.BASE_DIR, 'ml_models', 'moisture_scaler.pkl')

    try:
        result = subprocess.run([
            sys.executable,
            script_path,
            data_path,
            output_path,
            feature_scaler_path,
            moisture_scaler_path
        ], capture_output=True, text=True)
        
        return {
            'stdout': result.stdout,
            'stderr': result.stderr,
            'returncode': result.returncode
        }
    except Exception as e:
        return {'error': str(e)}
