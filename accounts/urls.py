from django.urls import path
from . import views
from django.contrib.auth import views as auth_views

urlpatterns = [
    path('', views.home, name='home'),
    path('register/', views.register, name='register'),
    path('login/', views.user_login, name='login'),
    path('logout/', views.user_logout, name='logout'),
    path('dashboard/', views.dashboard, name='dashboard'),
    path('profile/', views.profile, name='profile'),
    path('predict/', views.predict_soil, name='predict_soil'),
    path('history/', views.prediction_history, name='prediction_history'),
    path('download_report/', views.download_prediction_report, name='download_prediction_report'),
    path('download_report_pdf/', views.download_prediction_report_pdf, name='download_prediction_report_pdf'),
    path('analytics/', views.analytics_dashboard, name='analytics'),
    path('upload_model/', views.upload_model, name='upload_model'),
    path('view_models/', views.view_models, name='view_models'),
    path('activate_model/<int:model_id>/', views.activate_model, name='activate_model'),
    path('deactivate_model/<int:model_id>/', views.deactivate_model, name='deactivate_model'),
    path('retrain_model/<int:model_id>/', views.retrain_model, name='retrain_model'),
    # Password reset URLs
    path('password_reset/', auth_views.PasswordResetView.as_view(), name='password_reset'),
    path('password_reset/done/', auth_views.PasswordResetDoneView.as_view(), name='password_reset_done'),
    path('reset/<uidb64>/<token>/', auth_views.PasswordResetConfirmView.as_view(), name='password_reset_confirm'),
    path('reset/done/', auth_views.PasswordResetCompleteView.as_view(), name='password_reset_complete'),
    path('send_reset_code/', views.send_reset_code, name='send_reset_code'),
    path('verify_reset_code/', views.verify_reset_code, name='verify_reset_code'),
    path('set_new_password/', views.set_new_password, name='set_new_password'),
    path('alerts/mark_read/<int:alert_id>/', views.mark_alert_read, name='mark_alert_read'),
    path('prediction/<int:pk>/delete/', views.delete_prediction, name='delete_prediction'),
] 