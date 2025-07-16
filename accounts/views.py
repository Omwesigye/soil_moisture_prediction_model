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

User = get_user_model()

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

def generate_irrigation_recommendation(moisture):
    if moisture < 15:
        return "Irrigate"
    elif 15 <= moisture < 30:
        return "Reduce Irrigation"
    elif 30 <= moisture < 50:
        return "None"
    elif 50 <= moisture < 65:
        return "Reduce Irrigation"
    else:
        return "None"

@login_required
def dashboard(request):
    # Get latest prediction and generate recommendation
    latest_prediction = PredictionRecord.objects.filter(user=request.user).order_by('-created_at').first()
    irrigation_recommendation = None
    latest_moisture = None
    
    if latest_prediction:
        latest_moisture = latest_prediction.predicted_moisture
        irrigation_recommendation = generate_irrigation_recommendation(latest_moisture)
        
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

def get_latest_model():
    latest = MLModel.objects.order_by('-uploaded_at').first()
    if latest:
        return joblib.load(latest.model_file.path)
    return None

def get_status(moisture):
    if moisture < 15:
        return "Critical Low", "Irrigate"
    elif 15 <= moisture < 30:
        return "Dry", "Reduce Irrigation"
    elif 30 <= moisture < 50:
        return "Normal", "None"
    elif 50 <= moisture < 65:
        return "Wet", "Reduce Irrigation"
    else:
        return "Critical High", "None"

def predict_soil(request):
    if request.method == 'POST':
        input_method = request.POST.get('input_method')
        model = get_latest_model()
        if model is None:
            return render(request, 'accounts/predict_form.html', {'error': 'No model uploaded yet.'})

        if input_method == 'form':
            try:
                temperature_celcius = float(request.POST.get('temperature_celcius'))
                humidity_percent = float(request.POST.get('humidity_percent'))
                battery_voltage = float(request.POST.get('battery_voltage'))
                hour = int(request.POST.get('hour'))
                day = int(request.POST.get('day'))
                month = int(request.POST.get('month'))
                weekday = int(request.POST.get('weekday'))
                location_encoded = int(request.POST.get('location_encoded'))
                X = [[temperature_celcius, humidity_percent, battery_voltage, hour, day, month, weekday, location_encoded]]
                predicted_moisture = model.predict(X)[0]
                status, recommendation = get_status(predicted_moisture)
                # Save to database
                record = PredictionRecord.objects.create(  # type: ignore
                    user=request.user,
                    temperature_celcius=temperature_celcius,
                    humidity_percent=humidity_percent,
                    battery_voltage=battery_voltage,
                    hour=hour,
                    day=day,
                    month=month,
                    weekday=weekday,
                    location_encoded=location_encoded,
                    predicted_moisture=predicted_moisture,
                    status=status,
                    recommendation=recommendation
                )
                if request.user.role == 'farmer' and status in ['Critical Low', 'Low']:
                    pass
                return render(request, 'accounts/predict_result.html', {
                    'prediction': predicted_moisture,
                    'status': status,
                    'recommendation': recommendation
                })
            except Exception as e:
                return render(request, 'accounts/predict_form.html', {'error': f'Invalid input: {e}'})

        elif input_method == 'csv':
            csv_file = request.FILES.get('csv_file')
            if not csv_file:
                return render(request, 'accounts/predict_form.html', {'error': 'Please upload a CSV file.'})
            try:
                df = pd.read_csv(csv_file)
                required_columns = ['temperature_celcius', 'humidity_percent', 'battery_voltage', 'hour', 'day', 'month', 'weekday', 'location_encoded']
                if not all(col in df.columns for col in required_columns):
                    return render(request, 'accounts/predict_form.html', {'error': 'CSV must contain columns: ' + ', '.join(required_columns)})
                X = df[required_columns].values
                predictions = model.predict(X)
                statuses = []
                recommendations = []
                for i, moisture in enumerate(predictions):
                    status, recommendation = get_status(moisture)
                    statuses.append(status)
                    recommendations.append(recommendation)
                    # Save each row to database
                    record = PredictionRecord.objects.create(  # type: ignore
                        user=request.user,
                        temperature_celcius=df.iloc[i]['temperature_celcius'],
                        humidity_percent=df.iloc[i]['humidity_percent'],
                        battery_voltage=df.iloc[i]['battery_voltage'],
                        hour=df.iloc[i]['hour'],
                        day=df.iloc[i]['day'],
                        month=df.iloc[i]['month'],
                        weekday=df.iloc[i]['weekday'],
                        location_encoded=df.iloc[i]['location_encoded'],
                        predicted_moisture=moisture,
                        status=status,
                        recommendation=recommendation
                    )
                    if request.user.role == 'farmer' and status in ['Critical Low', 'Low']:
                        pass
                results = df.copy()
                results['predicted_moisture'] = predictions
                results['status'] = statuses
                results['recommendation'] = recommendations
                table_html = results.to_html(index=False, classes='prediction-table', border=0)
                return render(request, 'accounts/predict_result.html', {'prediction_table': table_html})
            except Exception as e:
                return render(request, 'accounts/predict_form.html', {'error': f'Error processing CSV: {e}'})

        else:
            return render(request, 'accounts/predict_form.html', {'error': 'Invalid input method.'})

    return render(request, 'accounts/predict_form.html')

@login_required
def prediction_history(request):
    records = PredictionRecord.objects.filter(user=request.user).order_by('-created_at')  # type: ignore
    # Filtering
    start_date = request.GET.get('start_date')
    end_date = request.GET.get('end_date')
    status = request.GET.get('status')
    recommendation = request.GET.get('recommendation')
    location_encoded = request.GET.get('location_encoded')
    if start_date:
        records = records.filter(created_at__date__gte=start_date)
    if end_date:
        records = records.filter(created_at__date__lte=end_date)
    if status and status != 'all':
        records = records.filter(status=status)
    if recommendation and recommendation != 'all':
        records = records.filter(recommendation=recommendation)
    if location_encoded and location_encoded != 'all':
        records = records.filter(location_encoded=location_encoded)
    # For dropdowns
    all_statuses = PredictionRecord.objects.filter(user=request.user).values_list('status', flat=True).distinct()  # type: ignore
    all_recommendations = PredictionRecord.objects.filter(user=request.user).values_list('recommendation', flat=True).distinct()  # type: ignore
    all_location_encoded = PredictionRecord.objects.filter(user=request.user).values_list('location_encoded', flat=True).distinct()  # type: ignore
    return render(request, 'accounts/prediction_history.html', {
        'records': records,
        'start_date': start_date or '',
        'end_date': end_date or '',
        'status': status or 'all',
        'all_statuses': all_statuses,
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
    low_moisture_count = all_records.filter(status='Low').count()
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
    if request.method == 'POST':
        # Option 1: Use uploaded file
        if 'training_file' in request.FILES:
            file = request.FILES['training_file']
            file_path = default_storage.save(f'ml_models/tmp/{file.name}', ContentFile(file.read()))
            data_path = os.path.join(settings.MEDIA_ROOT, file_path)
        else:
            # Option 2: Use previous data
            data_path = os.path.join(settings.BASE_DIR, 'ml_models', 'training_data.csv')
        output_path = model.model_file.path
        # Start Celery task
        task = retrain_model_task.delay(model.id, data_path, output_path)
        return redirect(f"{reverse('retrain_model', args=[model.id])}?task_id={task.id}")
    # If task_id is present, show status/logs
    if task_id:
        result = AsyncResult(task_id)
        if result.ready():
            log = result.result
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
