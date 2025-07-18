from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import UserRegistrationForm, UserLoginForm, UserProfileUpdateForm
from django.contrib import messages
from django.contrib.auth.decorators import login_required
import joblib
from .models import MLModel, PredictionRecord
import pandas as pd
from django.conf import settings
import random
from django.contrib.auth.hashers import make_password
from django.core.mail import send_mail
from .models import PasswordResetCode
from django.contrib.auth import views as auth_views
from django.contrib.auth import get_user_model
from django.http import HttpResponse
from django.contrib.auth.decorators import login_required
import pandas as pd
from datetime import datetime, timedelta
from django.template.loader import render_to_string
from weasyprint import HTML
from django.db.models.functions import TruncDate
from django.db.models import Avg
import datetime
from django.contrib.admin.views.decorators import staff_member_required
from django.contrib.admin import site
from django.shortcuts import redirect
from .forms import MLModelUploadForm
from django.shortcuts import get_object_or_404
import subprocess
import sys
import os
from .tasks import retrain_model_task
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from celery.result import AsyncResult
from django.urls import reverse
from django.http import JsonResponse
from .models import Alert
from django.views.decorators.http import require_POST
import numpy as np
from tensorflow import keras
import warnings
warnings.filterwarnings("ignore")
import glob

User = get_user_model()

# Paths
MODEL_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'soil_moisture_model.keras')
FEATURE_SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'feature_scaler.pkl')
MOISTURE_SCALER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'moisture_scaler.pkl')
ACTION_ENCODER_PATH = os.path.join(settings.BASE_DIR, 'ml_models', 'action_encoder.pkl')

# Remove global model/artifact loading
# model = keras.models.load_model(MODEL_PATH)
# feature_scaler = joblib.load(FEATURE_SCALER_PATH)
# moisture_scaler = joblib.load(MOISTURE_SCALER_PATH)
# action_encoder = joblib.load(ACTION_ENCODER_PATH)

def load_artifacts():
    import os
    import joblib
    from accounts.models import MLModel
    from django.core.exceptions import ObjectDoesNotExist

    try:
        # Try to get the active model from the database
        try:
            active_model = MLModel.objects.get(is_active=True)
            model_path = active_model.model_file.path
            if not os.path.exists(model_path):
                raise FileNotFoundError
        except (ObjectDoesNotExist, FileNotFoundError):
            # Fallback: use the latest .keras file in ml_models directory
            model_files = glob.glob(os.path.join(settings.BASE_DIR, 'ml_models', '*.keras'))
            if not model_files:
                raise RuntimeError("No model file found in ml_models directory.")
            model_path = max(model_files, key=os.path.getctime)  # Most recently created

        if not os.path.exists(FEATURE_SCALER_PATH):
            raise FileNotFoundError(f"Feature scaler not found: {FEATURE_SCALER_PATH}")
        if not os.path.exists(MOISTURE_SCALER_PATH):
            raise FileNotFoundError(f"Moisture scaler not found: {MOISTURE_SCALER_PATH}")
        if not os.path.exists(ACTION_ENCODER_PATH):
            raise FileNotFoundError(f"Action encoder not found: {ACTION_ENCODER_PATH}")

        model = keras.models.load_model(model_path)
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)
        moisture_scaler = joblib.load(MOISTURE_SCALER_PATH)
        action_encoder = joblib.load(ACTION_ENCODER_PATH)
        return model, feature_scaler, moisture_scaler, action_encoder
    except Exception as e:
        raise RuntimeError(f"Error loading ML model or artifacts: {e}")

# Create your views here.

def home(request):
    if request.user.is_authenticated:
        return redirect('dashboard')
    return render(request, 'accounts/home.html')

def send_moisture_alert(user, predicted_moisture, recommendation):
    """Send email and create in-app alert for low moisture levels"""
    threshold = 30  # Simple threshold for all crops
    
    if predicted_moisture < threshold:
        # Send email notification
        try:
            send_mail(
                'Soil Moisture Alert',
                f'Moisture level has dropped to {predicted_moisture}%. {recommendation}',
                'from@example.com',
                [user.email],
                fail_silently=True,
            )
        except Exception as e:
            print(f"Email sending failed: {e}")
        
        # Create in-app alert
        Alert.objects.create(
            user=user,
            message=f"Low moisture alert: {predicted_moisture}% - {recommendation}"
        )

@login_required
def dashboard(request):
    # Get latest prediction and generate recommendation
    latest_prediction = PredictionRecord.objects.filter(user=request.user).order_by('-created_at').first()
    irrigation_recommendation = None
    latest_moisture = None
    
    if latest_prediction:
        latest_moisture = latest_prediction.predicted_moisture
        irrigation_recommendation = latest_prediction.recommendation
        # Send notifications if moisture is low
        send_moisture_alert(request.user, latest_moisture, irrigation_recommendation)
    
    # Get recent alerts for the user
    recent_alerts = Alert.objects.filter(user=request.user, is_read=False).order_by('-created_at')[:5]
    
    # Moisture history for last 30 days
    from datetime import timedelta, date
    today = date.today()
    last_30 = today - timedelta(days=29)
    moisture_history = (
        PredictionRecord.objects.filter(user=request.user, created_at__date__gte=last_30)
        .annotate(created_day=TruncDate('created_at'))
        .values('created_day')
        .annotate(avg_moisture=Avg('predicted_moisture'))
        .order_by('created_day')
    )
    context = {
        'latest_prediction': latest_prediction,
        'irrigation_recommendation': irrigation_recommendation,
        'latest_moisture': latest_moisture,
        'recent_alerts': recent_alerts,
    }
    context['moisture_history_labels'] = [str(d['created_day']) for d in moisture_history]
    context['moisture_history_values'] = [d['avg_moisture'] for d in moisture_history]
    return render(request, 'accounts/dashboard.html', context)

@login_required
def profile(request):
    user = request.user
    if request.method == 'POST':
        form = UserProfileUpdateForm(request.POST, instance=user)
        if form.is_valid():
            form.save()
            return render(request, 'accounts/profile.html', {'form': form, 'success': True})
    else:
        form = UserProfileUpdateForm(instance=user)
    return render(request, 'accounts/profile.html', {'form': form})

def register(request):
    if request.method == 'POST':
        form = UserRegistrationForm(request.POST)
        if form.is_valid():
            user = form.save()
            email = form.cleaned_data['email']
            send_mail(
                subject='Welcome to Soil Monitoring System',
                message='Hello, you have successfully registered.',
                from_email=settings.DEFAULT_FROM_EMAIL,
                recipient_list=[email],
                fail_silently=False,
            )
            messages.success(request, 'Registration successful! Please log in.')
            return redirect('login')
    else:
        form = UserRegistrationForm()
    return render(request, 'accounts/register.html', {'form': form})

def send_reset_code(request):
    if request.method == 'POST':
        email = request.POST['email']
        try:
            user = User.objects.get(email=email)
        except User.DoesNotExist:
            messages.error(request, "No user with that email.")
            return redirect('send_reset_code')

        code = str(random.randint(100000, 999999))
        PasswordResetCode.objects.create(email=email, code=code)

        send_mail(
            'Password Reset Code',
            f'Your password reset code is: {code}',
            settings.DEFAULT_FROM_EMAIL,
            [email],
            fail_silently=False,
        )

        request.session['reset_email'] = email
        return redirect('verify_reset_code')

    return render(request, 'accounts/send_reset_code.html')


def verify_reset_code(request):
    if request.method == 'POST':
        entered_code = request.POST['code']
        email = request.session.get('reset_email')

        try:
            record = PasswordResetCode.objects.filter(email=email).latest('created_at')
        except PasswordResetCode.DoesNotExist:
            messages.error(request, "Invalid or expired code.")
            return redirect('send_reset_code')

        if record.code == entered_code:
            request.session['code_verified'] = True
            return redirect('set_new_password')
        else:
            messages.error(request, "Incorrect code.")

    return render(request, 'accounts/verify_code.html')


def set_new_password(request):
    if not request.session.get('code_verified'):
        return redirect('send_reset_code')

    if request.method == 'POST':
        password = request.POST['password']
        email = request.session.get('reset_email')
        user = User.objects.get(email=email)
        user.password = make_password(password)
        user.save()

        request.session.flush()
        messages.success(request, "Password updated. You can now log in.")
        return redirect('login')

    return render(request, 'accounts/set_new_password.html')
def user_login(request):
    if request.method == 'POST':
        form = UserLoginForm(request.POST)
        if form.is_valid():
            email = form.cleaned_data['email']
            password = form.cleaned_data['password']
            user = authenticate(request, email=email, password=password)
            if user is not None:
                login(request, user)
                if user.is_staff or user.is_superuser or getattr(user, 'role', None) == 'admin':
                    return redirect('/admin/')
                return redirect('dashboard')
            else:
                messages.error(request, 'Invalid email or password')
    else:
        form = UserLoginForm()
    return render(request, 'accounts/login.html', {'form': form})

def user_logout(request):
    logout(request)
    return redirect('home')

def admin_logout(request):
    logout(request)
    return redirect('home')

def predict_soil_moisture(sample_data: dict):
    # Load model and artifacts
    model, feature_scaler, moisture_scaler, irrigation_encoder = load_artifacts()
    # Use the keys from sample_data to build the input
    feature_order = ['temperature_celcius', 'humidity_percent', 'battery_voltage', 'hour', 'day', 'month', 'weekday', 'location_encoded']
    sample_input = np.array([[sample_data[feat] for feat in feature_order]])
    sample_input_scaled = feature_scaler.transform(sample_input)
    
    # Predict
    output = model.predict(sample_input_scaled)
    pred_moisture = output['moisture_level']
    pred_irrigation = output['irrigation_action']
    
    # Convert moisture back from [0,1] to [0,100]
    moisture_value = moisture_scaler.inverse_transform(pred_moisture.reshape(1, -1))[0][0]
    
    # Decode irrigation action label
    irrigation_index = np.argmax(pred_irrigation[0])
    irrigation_action = irrigation_encoder.inverse_transform([irrigation_index])[0]
    
    return moisture_value, irrigation_action

@login_required
def predict_soil(request):
    if request.method == 'POST':
        try:
            # Build sample_data dict from form fields
            sample_data = {
                'temperature_celcius': float(request.POST['temperature_celcius']),
                'humidity_percent': float(request.POST['humidity_percent']),
                'battery_voltage': float(request.POST['battery_voltage']),
                'hour': int(request.POST['hour']),
                'day': int(request.POST['day']),
                'month': int(request.POST['month']),
                'weekday': int(request.POST['weekday']),
                'location_encoded': int(request.POST['location_encoded']),
            }
            print('Sample data for model:', sample_data)
            try:
                predicted_moisture, irrigation_action = predict_soil_moisture(sample_data)
            except Exception as e:
                return render(request, 'accounts/predict_form.html', {'error': f'Model or artifacts missing/corrupt: {e}'})
            # Store prediction in the database
            from accounts.models import PredictionRecord
            PredictionRecord.objects.create(
                user=request.user,
                temperature_celcius=sample_data['temperature_celcius'],
                humidity_percent=sample_data['humidity_percent'],
                battery_voltage=sample_data['battery_voltage'],
                hour=sample_data['hour'],
                day=sample_data['day'],
                month=sample_data['month'],
                weekday=sample_data['weekday'],
                location_encoded=sample_data['location_encoded'],
                predicted_moisture=predicted_moisture,
                recommendation=irrigation_action
            )
            return render(request, 'accounts/predict_result.html', {
                'predicted_moisture': predicted_moisture,
                'irrigation_action': irrigation_action
            })
        except Exception as e:
            print("Prediction error:", e)
            return render(request, 'accounts/predict_form.html', {'error': f'Invalid input: {e}'})
    return render(request, 'accounts/predict_form.html')

@login_required
def prediction_history(request):
    records = PredictionRecord.objects.filter(user=request.user).order_by('-created_at')  # type: ignore
    # Filtering
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    recommendation = request.GET.get('recommendation')
    location_encoded = request.GET.get('location_encoded')
    if start_date:
        records = records.filter(created_at__date__gte=start_date)
    if end_date:
        records = records.filter(created_at__date__lte=end_date)
    if recommendation and recommendation != 'all':
        records = records.filter(recommendation=recommendation)
    if location_encoded and location_encoded != 'all':
        records = records.filter(location_encoded=location_encoded)
    # For dropdowns
    all_recommendations = PredictionRecord.objects.filter(user=request.user).values_list('recommendation', flat=True).distinct()  # type: ignore
    all_location_encoded = PredictionRecord.objects.filter(user=request.user).values_list('location_encoded', flat=True).distinct()  # type: ignore
    return render(request, 'accounts/prediction_history.html', {
        'records': records,
        'start_date': start_date or '',
        'end_date': end_date or '',
        'recommendation': recommendation or 'all',
        'all_recommendations': all_recommendations,
        'location_encoded': location_encoded or 'all',
        'all_location_encoded': all_location_encoded,
    })

@login_required
def download_prediction_report(request):
    period = request.GET.get('period', 'daily')
    user = request.user
    now = datetime.now()

    if period == 'daily':
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'weekly':
        start_date = now - timedelta(days=now.weekday())
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'monthly':
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

    records = PredictionRecord.objects.filter(user=user, created_at__gte=start_date)
    if not records.exists():
        return HttpResponse('No records found for this period.', content_type='text/plain')

    # Prepare data for DataFrame
    data = list(records.values())
    df = pd.DataFrame(data)
    # Convert all datetime columns to timezone-naive
    for col in df.select_dtypes(include=['datetimetz']).columns:
        df[col] = df[col].dt.tz_localize(None)

    # Optional: Format/rename columns if needed

    # Create Excel file in memory
    response = HttpResponse(content_type='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')
    filename = f'prediction_report_{period}_{now.strftime("%Y%m%d_%H%M%S")}.xlsx'
    response['Content-Disposition'] = f'attachment; filename={filename}'
    with pd.ExcelWriter(response, engine='openpyxl') as writer:
        df.to_excel(writer, index=False, sheet_name='Predictions')
    return response

@login_required
def download_prediction_report_pdf(request):
    period = request.GET.get('period', 'daily')
    user = request.user
    now = datetime.datetime.now()

    if period == 'daily':
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'weekly':
        start_date = now - timedelta(days=now.weekday())
        start_date = start_date.replace(hour=0, minute=0, second=0, microsecond=0)
    elif period == 'monthly':
        start_date = now.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    else:
        start_date = now.replace(hour=0, minute=0, second=0, microsecond=0)

    records = PredictionRecord.objects.filter(user=user, created_at__gte=start_date)
    if not records.exists():
        return HttpResponse('No records found for this period.', content_type='text/plain')

    # Render HTML template with records
    html_string = render_to_string('accounts/prediction_report_pdf.html', {
        'records': records,
        'user': user,
        'period': period,
        'now': now,
        'start_date': start_date,
    })
    html = HTML(string=html_string)
    pdf_file = html.write_pdf()

    response = HttpResponse(pdf_file, content_type='application/pdf')
    filename = f'prediction_report_{period}_{now.strftime("%Y%m%d_%H%M%S")}.pdf'
    response['Content-Disposition'] = f'attachment; filename={filename}'
    return response

@login_required
def analytics_dashboard(request):
    # Fetch the latest 50 records for real-time chart
    recent_records = PredictionRecord.objects.filter(user=request.user).order_by('-created_at')[:50][::-1]
    # Calculate summary statistics
    all_records = PredictionRecord.objects.filter(user=request.user)
    avg_moisture = all_records.aggregate(pd_avg=Avg('predicted_moisture'))['pd_avg']
    # Daily trend (last 7 days)
    today = datetime.date.today()
    last_week = today - datetime.timedelta(days=6)
    daily_trends = (
        all_records.filter(created_at__date__gte=last_week)
        .annotate(created_day=TruncDate('created_at'))
        .values('created_day')
        .annotate(avg_moisture=Avg('predicted_moisture'))
        .order_by('created_day')
    )
    # Risk warnings (e.g., count of low moisture predictions)
    low_moisture_count = all_records.filter(recommendation='Irrigate').count() # Changed from status to recommendation
    # Predicted moisture values for line chart (last 20 predictions)
    predicted_records = all_records.order_by('-created_at')[:20][::-1]
    predicted_labels = [rec.created_at.strftime('%Y-%m-%d %H:%M') for rec in predicted_records]
    predicted_values = [rec.predicted_moisture for rec in predicted_records]
    context = {
        'recent_records': recent_records,  # Real-time data
        'avg_moisture': avg_moisture,     # Summary
        'daily_trends': list(daily_trends), # Daily trends
        'low_moisture_count': low_moisture_count, # Risk warnings
        'predicted_labels': predicted_labels,
        'predicted_values': predicted_values,
    }
    return render(request, 'accounts/analytics.html', context)

@staff_member_required
def upload_model(request):
    if request.method == 'POST':
        form = MLModelUploadForm(request.POST, request.FILES)
        if form.is_valid():
            form.save()
            messages.success(request, "Model uploaded successfully!")
            return redirect('view_models')
    else:
        form = MLModelUploadForm()
    return render(request, 'accounts/upload_model.html', {'form': form})

@staff_member_required
def view_models(request):
    models = MLModel.objects.order_by('-uploaded_at')
    return render(request, 'accounts/view_models.html', {'models': models})

@staff_member_required
def activate_model(request, model_id):
    model = get_object_or_404(MLModel, id=model_id)
    MLModel.objects.update(is_active=False)  # Deactivate all
    model.is_active = True
    model.save()
    messages.success(request, f"Model '{model.name}' activated.")
    return redirect('view_models')

@staff_member_required
def deactivate_model(request, model_id):
    model = get_object_or_404(MLModel, id=model_id)
    model.is_active = False
    model.save()
    messages.success(request, f"Model '{model.name}' deactivated.")
    return redirect('view_models')

@staff_member_required
def retrain_model(request, model_id):
    model = get_object_or_404(MLModel, id=model_id)
    log = None
    task_id = request.GET.get('task_id')
    import os
    from tensorflow import keras
    if request.method == 'POST':
        # Option 1: Use uploaded file
        if 'training_file' in request.FILES:
            file = request.FILES['training_file']
            file_path = default_storage.save(f'ml_models/tmp/{file.name}', ContentFile(file.read()))
            data_path = os.path.join(settings.MEDIA_ROOT, file_path)
        else:
            # Option 2: Use the correct dataset path
            data_path = os.path.join(settings.BASE_DIR, 'ml_models', 'cleaned_soil_moisture_dataset.csv')
        model_dir = os.path.dirname(model.model_file.path)
        # Start Celery task
        task = retrain_model_task.delay(model.id, data_path, model_dir)
        return redirect(f"{reverse('retrain_model', args=[model.id])}?task_id={task.id}")
    # If task_id is present, show status/logs
    if task_id:
        result = AsyncResult(task_id)
        if result.ready():
            log = result.result
            # Post-retrain check: verify model file
            model_path = model.model_file.path
            if os.path.exists(model_path):
                try:
                    keras.models.load_model(model_path)
                    messages.success(request, "Model retrained, saved, and ready for predictions!")
                except Exception as e:
                    messages.error(request, f"Model file exists but is invalid: {e}")
            else:
                messages.error(request, "Model file was not created after retraining.")
    return render(request, 'accounts/retrain_model.html', {'model': model, 'log': log, 'task_id': task_id})

def custom_admin_login(request):
    # If user is authenticated and is staff/admin, redirect to admin index
    if request.user.is_authenticated and (request.user.is_staff or request.user.is_superuser or getattr(request.user, 'role', None) == 'admin'):
        return redirect('/admin/')
    # Otherwise, redirect to your custom login page
    return redirect('login')

@require_POST
@login_required
def mark_alert_read(request, alert_id):
    alert = get_object_or_404(Alert, id=alert_id, user=request.user)
    alert.is_read = True
    alert.save()
    return redirect('dashboard')

from django.shortcuts import render, get_object_or_404, redirect
from .models import PredictionRecord
from .forms import PredictionRecordForm

def delete_prediction(request, pk):
    prediction = get_object_or_404(PredictionRecord, pk=pk, user=request.user)
    if request.method == 'POST':
        prediction.delete()
        return redirect('prediction_history')
    return redirect('prediction_history')
