from django.contrib.auth.models import AbstractBaseUser, PermissionsMixin, BaseUserManager
from django.db import models
from django.conf import settings
from django.utils import timezone
import uuid
from django.contrib.auth import get_user_model
from .custom_storage import MLModelStorage

# Create your models here.

class CustomUserManager(BaseUserManager):
    def create_user(self, email, name, telno, location, password=None, **extra_fields):
        if not email:
            raise ValueError('The Email field must be set')
        email = self.normalize_email(email)
        user = self.model(email=email, name=name, telno=telno, location=location, **extra_fields)
        user.set_password(password)
        user.save(using=self._db)
        return user

    def create_superuser(self, email, name, telno, location, password=None, **extra_fields):
        extra_fields.setdefault('is_staff', True)
        extra_fields.setdefault('is_superuser', True)
        extra_fields['role'] = 'admin'
        return self.create_user(email, name, telno, location, password, **extra_fields)

class CustomUser(AbstractBaseUser, PermissionsMixin):
    email = models.EmailField(unique=True, max_length=191)
    name = models.CharField(max_length=30)
    telno = models.CharField(max_length=20)
    location = models.CharField(max_length=20)
    ROLE_CHOICES = [
        ('admin', 'Admin'),
        ('technician', 'Technician'),
        ('farmer', 'Farmer'),
    ]
    role = models.CharField(max_length=20, choices=ROLE_CHOICES, default='farmer')
    is_active = models.BooleanField(default=True)
    is_staff = models.BooleanField(default=False)

    objects = CustomUserManager()

    USERNAME_FIELD = 'email'
    REQUIRED_FIELDS = ['name', 'telno', 'location']

    def __str__(self):
        return self.email

class MLModel(models.Model):
    name = models.CharField(max_length=100, default="Soil Model")
    model_file = models.FileField(upload_to='', storage=MLModelStorage())  # Store directly in ml_models/
    uploaded_at = models.DateTimeField(auto_now_add=True)
    is_active = models.BooleanField(default=False)

    def __str__(self):
        return self.name

class PredictionRecord(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE)
    temperature_celcius = models.FloatField()
    humidity_percent = models.FloatField()
    battery_voltage = models.FloatField()
    hour = models.IntegerField()
    day = models.IntegerField()
    month = models.IntegerField()
    weekday = models.IntegerField()
    location_encoded = models.IntegerField()
    predicted_moisture = models.FloatField()
    recommendation = models.CharField(max_length=64)
    created_at = models.DateTimeField(auto_now_add=True)
    def __str__(self):
        return f"Prediction by {self.user} on {self.created_at}"

class PasswordResetCode(models.Model):
    email = models.EmailField(max_length=191)
    code = models.CharField(max_length=6)
    created_at = models.DateTimeField(default=timezone.now)
    def _str_(self):
        return f"{self.email} - {self.code}"

class Alert(models.Model):
    user = models.ForeignKey(get_user_model(), on_delete=models.CASCADE, related_name='alerts')
    message = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)
    is_read = models.BooleanField(default=False)

    def __str__(self):
        return f"Alert for {self.user} at {self.created_at}: {self.message[:30]}"
