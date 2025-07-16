from django.contrib import admin
from django.urls import path
from django.utils.decorators import method_decorator
from django.contrib.auth.decorators import login_required
from django.contrib.admin import AdminSite
from accounts.models import PredictionRecord, CustomUser, MLModel
from .tasks import retrain_model_task
from django.utils.html import format_html
from django.urls import reverse
from django.http import HttpResponseRedirect
from django.contrib import messages
from django.template.response import TemplateResponse
from celery.result import AsyncResult
from django.shortcuts import render

class CustomAdminSite(AdminSite):
    site_header = 'Soil Moisture Detector Admin'
    site_title = 'Soil Moisture Admin'
    index_title = 'Dashboard'

    @method_decorator(login_required)
    def index(self, request, extra_context=None):
        total_predictions = PredictionRecord.objects.count()
        active_farmers = CustomUser.objects.filter(role='farmer', is_active=True).count()
        critical_alerts = PredictionRecord.objects.filter(status__icontains='Critical').count()
        num_technicians = CustomUser.objects.filter(role='technician', is_active=True).count()
        num_models = MLModel.objects.count()
        recent_predictions = PredictionRecord.objects.select_related('user').order_by('-created_at')[:5]
        if extra_context is None:
            extra_context = {}
        extra_context.update({
            'total_predictions': total_predictions,
            'active_farmers': active_farmers,
            'critical_alerts': critical_alerts,
            'num_technicians': num_technicians,
            'num_models': num_models,
            'recent_predictions': recent_predictions,
        })
        return super().index(request, extra_context=extra_context)

custom_admin_site = CustomAdminSite(name='custom_admin')

class MLModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'uploaded_at', 'is_active', 'retrain_link')
    actions = ['retrain_selected_models']

    def get_urls(self):
        urls = super().get_urls()
        custom_urls = [
            path(
                '<int:model_id>/retrain/',
                self.admin_site.admin_view(self.retrain_model_view),
                name='accounts_mlmodel_retrain',
            ),
            path(
                '<int:model_id>/retrain_status/<task_id>/',
                self.admin_site.admin_view(self.retrain_status_view),
                name='accounts_mlmodel_retrain_status',
            ),
        ]
        return custom_urls + urls

    def retrain_link(self, obj):
        url = reverse('admin:accounts_mlmodel_retrain', args=[obj.id])
        return format_html('<a class="button" href="{}">Retrain</a>', url)
    retrain_link.short_description = 'Retrain Model'
    retrain_link.allow_tags = True

    def retrain_model_view(self, request, model_id):
        model = MLModel.objects.get(id=model_id)
        data_path = 'ml_models/training_data.csv'  # Update as needed
        output_path = model.model_file.path
        task = retrain_model_task.delay(model.id, data_path, output_path)
        messages.success(
            request,
            f'Retraining started for model "{model.name}". '
            f'<a href="{reverse("admin:accounts_mlmodel_retrain_status", args=[model.id, task.id])}">Check status</a>',
            extra_tags='safe'
        )
        return HttpResponseRedirect(request.META.get('HTTP_REFERER', '/admin/'))

    def retrain_status_view(self, request, model_id, task_id):
        result = AsyncResult(task_id)
        context = {
            'model_id': model_id,
            'task_id': task_id,
            'status': result.status,
            'result': result.result,
        }
        return render(request, 'admin/retrain_status.html', context)

    def retrain_selected_models(self, request, queryset):
        for model in queryset:
            data_path = 'ml_models/training_data.csv'
            output_path = model.model_file.path
            retrain_model_task.delay(model.id, data_path, output_path)
        self.message_user(request, "Retraining started for selected models.")
    retrain_selected_models.short_description = "Retrain selected models"

# Register your models with the custom admin site
custom_admin_site.register(PredictionRecord)
custom_admin_site.register(CustomUser)
custom_admin_site.register(MLModel, MLModelAdmin)
