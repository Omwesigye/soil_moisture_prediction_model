from django import forms
from django.contrib.auth.forms import ReadOnlyPasswordHashField
from .models import CustomUser
from django.contrib.auth import authenticate
from .models import MLModel
from .models import PredictionRecord

class UserRegistrationForm(forms.ModelForm):
    password1 = forms.CharField(label='Password', widget=forms.PasswordInput)
    password2 = forms.CharField(label='Confirm Password', widget=forms.PasswordInput)
    # role = forms.ChoiceField(choices=CustomUser.ROLE_CHOICES, label='Register as')  # Removed

    class Meta:
        model = CustomUser
        fields = ('name', 'email', 'telno', 'location')  # Removed 'role'

    def clean_password2(self):
        password1 = self.cleaned_data.get('password1')
        password2 = self.cleaned_data.get('password2')
        if password1 and password2 and password1 != password2:
            raise forms.ValidationError("Passwords don't match")
        return password2

    def save(self, commit=True):
        user = super().save(commit=False)
        user.set_password(self.cleaned_data["password1"])
        # user.role = self.cleaned_data["role"]  # Removed
        if commit:
            user.save()
        return user

class UserLoginForm(forms.Form):
    email = forms.EmailField()
    password = forms.CharField(widget=forms.PasswordInput)

    def clean(self):
        email = self.cleaned_data.get('email')
        password = self.cleaned_data.get('password')
        if email and password:
            user = authenticate(email=email, password=password)
            if not user:
                raise forms.ValidationError('Invalid email or password')
        return self.cleaned_data 

class UserProfileUpdateForm(forms.ModelForm):
    email = forms.EmailField(disabled=True, required=False, label='Email (cannot change)')
    class Meta:
        model = CustomUser
        fields = ('name', 'email', 'telno', 'location') 

class MLModelUploadForm(forms.ModelForm):
    class Meta:
        model = MLModel
        fields = ['name', 'model_file']
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields['model_file'].widget.attrs['accept'] = '.h5,.pkl,.joblib,*' 

class PredictionRecordForm(forms.ModelForm):
    class Meta:
        model = PredictionRecord
        fields = '__all__'  # or specify the fields you want to allow editing 