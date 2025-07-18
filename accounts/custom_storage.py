import os
from django.core.files.storage import FileSystemStorage
from django.conf import settings

class MLModelStorage(FileSystemStorage):
    def __init__(self, *args, **kwargs):
        # Store in <project_root>/ml_models/
        location = os.path.join(settings.BASE_DIR, 'ml_models')
        super().__init__(location=location, *args, **kwargs) 